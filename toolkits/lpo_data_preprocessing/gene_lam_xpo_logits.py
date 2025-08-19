import os
import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
import random

from utils.ddp_tools import *
from utils.common_utils import write_to_json, read_data
from utils.tokenizer import build_qwen_tokenizer


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input data")
    group.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )
    
    group.add_argument(
        "--loss_type",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
        default='sigmoid'
    )
    
    group.add_argument(
        "--ref_model_path",
        type=str,
        required=True,
        help="Path to input jsonl files or lmd archive(s) - if using multiple archives, put them in a comma separated "
        "list",
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )

    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )

    group = parser.add_argument_group(title="runtime")

    group.add_argument(
        "--epoch",
        type=int,
        default=1,
    )
    group.add_argument(
        "--seq_length",
        type=int,
        default=4096,
    )
    group.add_argument(
        "--batch_size",
        type=int,
        default=1024,
    )
    group.add_argument(
        "--save_chunk_size",
        type=int,
        default=50000,
    )
    group.add_argument(
        "--micro_batch_size",
        type=int,
        default=6,
    )
    group.add_argument(
        "--average_logprobs",
        action="store_true",
        default=False,
        help="whether to add weather info",
    )
    group.add_argument(
        "--group_num",
        type=int,
        default=1,
    )
    group.add_argument(
        "--skip", 
        action="store_true", 
        default=False)
    args = parser.parse_args()

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


class LAMXPOEncoder(object):
    def __init__(self, args, rank, local_rank):
        self.args = args
        
        self.rank = rank
        self.local_rank = local_rank

        self.max_tokens_len = 0
        
        self.micro_bsz = args.micro_batch_size
        self.chunk_size = args.save_chunk_size
        
        self.initializer()
        
    def initializer(self):
        self.tokenizer = build_qwen_tokenizer(self.args)
        self.eos_token = self.tokenizer.eos_id
        self.seq_length = self.args.seq_length
        self.model = AutoModelForCausalLM.from_pretrained(self.args.ref_model_path).bfloat16().to(self.local_rank)
        self.model = nn.parallel.DistributedDataParallel(self.model)
        self.model.eval()
        torch.cuda.empty_cache()
        print_rank_0('after load model:')
        print_rank_0(os.popen('nvidia-smi').read())
        
    def encode(self, item) -> list:
        if isinstance(item, str):
            item = json.loads(item)
        tokenized_data = self.tokenize(item=item)
        formatted_data = self.formatted_data(tokenized_data)
        parts = [formatted_data[i: i + self.micro_bsz] for i in range(0, len(formatted_data), self.micro_bsz)]
        samples_with_logprobs = []
        for samples in parts:
            samples_with_logprobs.extend(self.get_reference_model_logprobs(sample=samples))
        chosen_sample = samples_with_logprobs[0]
        reject_samples = samples_with_logprobs[1:]
        xpo_pair = []
        for rs in reject_samples:
            self.length_double_check(chosen_sample, rs)
            pair_item = {
                'chosen': chosen_sample,
                'reject': rs,
                'task': item['task_category']
            }
            xpo_pair.append(pair_item)
        
        return xpo_pair
    
    def length_double_check(self, chosen, reject):
        assert len(chosen['input_ids']) == len(reject['input_ids'])
        # assert len(chosen['loss_mask']) == len(reject['loss_mask'])
        # assert len(chosen['input_ids']) == len(chosen['loss_mask'])
        # assert len(reject['input_ids']) == len(reject['loss_mask'])
    
    def formatted_data(self, tokenized_data):
        prompt_tokens = tokenized_data['prompt_tokens']
        prompt_tokens_length = len(prompt_tokens)
        chosen_tokens = prompt_tokens + tokenized_data['chosen_answer_tokens']
        answer_tokens_length = len(tokenized_data['chosen_answer_tokens'])
        chosen_loss_mask = [0] * prompt_tokens_length + [1] * answer_tokens_length
        chosen_loss_mask_idx = [prompt_tokens_length, 
                                prompt_tokens_length + answer_tokens_length]
        chosen_sample = {
            'input_ids': chosen_tokens,
            'loss_mask': chosen_loss_mask,
            'loss_mask_idx': chosen_loss_mask_idx
        }

        formatted_samples = []
        for rat in tokenized_data['reject_answer_tokens']:
            reject_answer_length = len(rat)
            reject_loss_mask = [0] * prompt_tokens_length + [1] * reject_answer_length
            reject_tokens = prompt_tokens + rat
            assert len(reject_loss_mask) == len(reject_tokens)
            reject_loss_mask_idx = [prompt_tokens_length, 
                                    prompt_tokens_length + reject_answer_length]
            reject_sample = {
                'input_ids': reject_tokens,
                'loss_mask': reject_loss_mask,
                'loss_mask_idx': reject_loss_mask_idx
            }
            formatted_samples.append(reject_sample)
            
        all_samples = [chosen_sample] + formatted_samples

        return all_samples
        
    def tokenize(self, item: dict):
        prompt = item['model_input']
        prompt_tokens = self.tokenizer.tokenize(prompt)

        chosen_answer = item['better_context']
        chosen_answer_tokens = self.tokenizer.tokenize(chosen_answer)
        encoded_sample_pair = {
            "prompt_tokens": prompt_tokens,
            "chosen_answer_tokens": chosen_answer_tokens,
            "reject_answer_tokens": [],
            "task_category": item['task_category'],
        }
        for reject_candidate in item['candidate_set']:
            reject_answer_tokens = self.tokenizer.tokenize(reject_candidate)
            encoded_sample_pair['reject_answer_tokens'].append(reject_answer_tokens)
            
        return encoded_sample_pair
    
    def compute_logprobs(self, input_ids, loss_mask, average_log_prob=True):
        # for id, lmid in zip(input_ids[0], loss_mask[0]):
        #   print_rank_0(f'{id} {lmid}')
        # print_rank_0(self.tokenizer.detokenize(input_ids[0]))
        input_ids = torch.LongTensor(input_ids).to(self.model.device)
        loss_mask = torch.tensor(loss_mask)
        with torch.no_grad():
            all_logits = self.model(input_ids[:, :-1],
                                    use_cache=False).logits
            labels = input_ids[:, 1:]
            # loss_mask = loss_mask[:, :-1].to(self.model.device) 原始的结果
            loss_mask = loss_mask[:, 1:].to(self.model.device)
            per_token_logps = torch.gather(all_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        
        if average_log_prob or self.args.average_logprobs:
            log_probs = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
            # print_rank_0(per_token_logps[0, 800: 1000], loss_mask.sum(-1), log_probs)
        else:
            log_probs = (per_token_logps * loss_mask).sum(-1)
            
        log_probs = log_probs.tolist()
            
        return log_probs
    
    def is_in_loss_mask_range(self, idx, begin_idx, end_idx):
        if idx >= begin_idx and idx < end_idx:
            return True
        else:
            return False
        
    def build_loss_mask_from_idx(self, loss_mask_idx):
        loss_mask = []
        for lm_idx in loss_mask_idx:
            begin_idx = lm_idx[0]
            end_idx = lm_idx[1]
            loss_mask.append([1 if self.is_in_loss_mask_range(index, begin_idx, end_idx) else 0 
                              for index in range(end_idx)])
        
        return loss_mask
    
    def get_reference_model_logprobs(self, sample) -> list:
        is_average = False if self.args.loss_type != 'ipo' else True

        input_ids = [s['input_ids'] for s in sample]
        # print(f'Detokenized: {self.tokenizer.detokenize(input_ids)}\n{"*"*10}')
        # loss_mask = [s['loss_mask'] for s in sample]
        loss_mask_idx = [s['loss_mask_idx'] for s in sample]
        loss_mask = self.build_loss_mask_from_idx(loss_mask_idx)
        max_length = max([len(ele) for ele in input_ids])
        
        input_ids_padding = self._padding_to_seq_length(
            sequence=input_ids,
            padding_length=max_length,
            padding_token=self.tokenizer.eos_id)
        loss_mask_padding = self._padding_to_seq_length(
            sequence=loss_mask,
            padding_length=max_length,
            padding_token=0)
        logprobs = self.compute_logprobs(input_ids=input_ids_padding,
                                         loss_mask=loss_mask_padding,
                                         average_log_prob=is_average)
    
        sample_with_logprobs = []
        for ids, lm_idx, prob in zip(input_ids, loss_mask_idx, logprobs):
            ids_padding = self._padding_to_seq_length([ids], self.seq_length, self.tokenizer.eos_id)[0]
            ele = {
                'input_ids': ids_padding,
                'loss_mask_idx': lm_idx,
                'logprob': prob
            }
            sample_with_logprobs.append(ele)
        
        return sample_with_logprobs
    
    def _padding_to_seq_length(self, sequence, padding_length=None, padding_token=None):
        new_sequence = []
        for seq in sequence:
            new_sample_tokens = seq + [padding_token] * padding_length
            new_sequence.append(new_sample_tokens[:padding_length])
        
        return new_sequence
    
    def padding(self, sample_pair):
        new_chosen_sample = self._padding_to_seq_length(sample_pair['chosen'])
        new_reject_sample = self._padding_to_seq_length(sample_pair['reject'])
        
        new_sample_pair = {
            'chosen': new_chosen_sample,
            'reject': new_reject_sample
        }
        
        return new_sample_pair
    

def get_dataloader(datas, batch_size):
    sample_list = [data for data in datas]
    sampler = torch.utils.data.distributed.DistributedSampler(sample_list, 
                                                              shuffle=False)
    # Torch dataloader.
    dataloder = torch.utils.data.DataLoader(
        sample_list,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=xpo_collator,
    )
    
    return dataloder


def generate_xpo_data(xpo_encoder: LAMXPOEncoder, dataloder, epoch, output_dir):
    local_xpo_data = []
    chunk_index = 0
    for data_path in tqdm(dataloder,
                          desc=f'XPO-Processing-{xpo_encoder.rank}'):
        print(f'Rank{xpo_encoder.rank} loading data...')
        data_list = read_data(data_path=data_path[0], rank=xpo_encoder.rank)
        for i, batch_index in enumerate(range(0, len(data_list), xpo_encoder.args.batch_size)):
            batch = data_list[batch_index: batch_index + xpo_encoder.args.batch_size]
            for item in tqdm(batch, desc=f'Proc-{xpo_encoder.rank}-{i}'):
                try:
                    dpo_data = xpo_encoder.encode(item=item)
                except Exception as e:
                   print('Proc data error')
                   continue
                if dpo_data:
                    local_xpo_data.extend(dpo_data)
            
            torch.cuda.empty_cache()
            print_rank_0(os.popen('nvidia-smi').read())

            if len(local_xpo_data) >= xpo_encoder.chunk_size:
                save_chunk_size_data(local_xpo_data, xpo_encoder.rank, chunk_index, output_dir, epoch)
                chunk_index += 1
                local_xpo_data = []                                                 
    
    if len(local_xpo_data) > 0:
        save_chunk_size_data(local_xpo_data, xpo_encoder.rank, chunk_index, output_dir, epoch)

    return local_xpo_data
                            

def save_chunk_size_data(data, local_rank, chunk_index, output_dir, epoch=1, turn=0):
    xpo_dataset = []
    for _ in range(epoch):
        # random.shuffle(data)
        xpo_dataset.extend(data)

    output_file = os.path.join(output_dir, f'train_r{local_rank}_i{chunk_index}_t{turn}.json')
    write_to_json(data=xpo_dataset,
                  data_path=output_file)
    print(f'Rank{local_rank} total size = {len(xpo_dataset)}, saving to {output_file}')


def main():
    args = get_args()
    print(args)
    
    if args.skip:
        return

    setup()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    pid = os.getpid()
    print(f'current pid: {pid}, rank: {rank}, local rank: {local_rank}')
    
    print_rank_0('Making output directory...')
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            jsonl_path = os.path.join(args.output, 'jsonl_data')
            if not os.path.exists(jsonl_path):
                os.mkdir(jsonl_path)

    xpo_encoder = LAMXPOEncoder(args, rank, local_rank)
    if not os.path.exists(args.input):
        print_rank_0(f'{args.input} not exists, exit')
        return
    print_rank_0(f"process dpo_train start")
    print_rank_0(f'  > pre-tokenize dataset[dpo_train] start')
    
    all_pathes = [os.path.join(args.input, name) for name in os.listdir(args.input) 
                  if not os.path.isdir(os.path.join(args.input, name))]
    print_rank_0(all_pathes)
    # all_data = []
    # for file_path in all_pathes:
    #    data_list = read_data(data_path=file_path)
    #    all_data.extend(data_list)
    
    # dataloder = get_dataloader(datas=all_data, batch_size=args.batch_size)
    dataloder = get_dataloader(datas=all_pathes, batch_size=1)
    jsonl_path = os.path.join(args.output, 'jsonl_data')
    generate_xpo_data(xpo_encoder, dataloder, args.epoch, jsonl_path)
    
    cleanup()


if __name__ == '__main__':
    main()
