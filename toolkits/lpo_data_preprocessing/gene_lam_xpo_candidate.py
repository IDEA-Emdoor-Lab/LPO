import os
import json
import random

import argparse
from tqdm import tqdm
import torch
import copy

from utils.common_utils import read_data, write_to_json, merge_jsonl_files
from vllm import LLM, SamplingParams
from utils.tokenizer import build_qwen_tokenizer, QWenTokenizer
import multiprocessing
import time


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="runtime")

    group.add_argument('--input_dir', type=str, default="/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/20250214/org_data/lpo_train.jsonl")
    group.add_argument('--model_path', type=str, default="")
    group.add_argument('--tokenizer_path', type=str)
    group.add_argument('--output_dir', type=str, default="/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/20250214/lpo_candidate")
    group.add_argument('--candidate_num', type=int, default=5)
    group.add_argument('--gpus_per_proc', type=int, default=1)
    group.add_argument('--num_gpus', type=int, default=1)
    group.add_argument('--save_chunk_size', type=int, default=5000)
    group.add_argument('--seq_length', type=int, default=4096)
    group.add_argument("--only_merge", action="store_true", default=False)
    group.add_argument("--skip", action="store_true", default=False)
    group.add_argument(
        "--vocab-file", type=str, 
        default="/cognitive_comp/ccnl_common_data/wangrui/audio-text-models/qwen_models/Qwen2.5-7B-Codec0927-S204000-AEdivided100", 
        help="Path to the vocab file"
    )
    group.add_argument('--group_num',
                    type=int,
                    default=1,
                    help='group num')
    group.add_argument('--tokenizer_type', type=str, default="QWen")
    group.add_argument('--topp', type=float, default=1.0)
    group.add_argument('--temp', type=float, default=1.0)
    
    args = parser.parse_args()

    return args


def write_output(args, sub_result, sub_prompt_list, output_file):
    nc = args.candidate_num
    n_output_data = [sub_result[i: i + nc] for i in range(0, len(sub_result), nc)]
    assert len(n_output_data) == len(sub_prompt_list)
    for i, outputs in enumerate(n_output_data):
        for res in outputs:
            output_text = res.outputs[0].text.rstrip()
            if not output_text.endswith('<|im_end|><|endoftext|>'):
                output_text = output_text + '<|im_end|><|endoftext|>'
            sub_prompt_list[i]['candidate_set'].append(output_text)
    write_to_json(sub_prompt_list, output_file)


class Encoder(object):
    def __init__(self, args):
        self.args = args

        self.conv_template = "<|im_start|>{role_name}\n{message}<|im_end|><|endoftext|>\n"

        self.n_single_print = 0

        self.is_concat_print = True

        self.initializer()
        
    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer: QWenTokenizer = build_qwen_tokenizer(self.args)
        Encoder.seq_length = self.args.seq_length
    
    def convert_audioID2str(self, audio_tokens) -> str:
        audio_token_str = ''
        for audio_token in audio_tokens:
            audio_token_str += f'<|g0r0_{audio_token}|>'

        return audio_token_str
    
    def valid_task_type(self, task_name: str) -> bool:
        tasks = ['zeroshot_tts', 'text_sft', 'r1']
        for t in tasks:
            if t in task_name:
                return True
        
        return False

    def encode(self, item: dict):
        item = json.loads(item)
        task = item.get('task', 'unknown')
        # if not self.valid_task_type(task_name=task):
        #  print(task)
        #  return None
        
        if 'dialog' in item:
            dialoge = item['dialog']
        else:
            dialoge = item['dialoge']

        sample_in_str = ''

        # 增加role dialog膨胀的对话数据集
        samples = []
        dialoge = json.loads(dialoge) if isinstance(dialoge, str) else dialoge
        
        # print(f'dialogeis:{dialoge}', flush=True)
        for index, conv_turn in enumerate(dialoge):
            role_name = None
            role_type = None
            if 'type' in conv_turn:
                role_type = conv_turn['type']
            elif 'role' in conv_turn: 
                role_type = conv_turn['role']
    
            if 'name' in conv_turn:
                role_name = conv_turn['name']
            if role_name is None:
                role_name = role_type
            if role_type is None:
                raise ValueError(f'can not find role type!')

            if 'system' in role_type:
                role_name = 'system'
                cal_loss = False
            elif role_type == 'assistant':
                cal_loss = True
            elif role_type == 'user':
                cal_loss = False
            else:
                raise ValueError(f'{role_type} not supported yet!')
            if 'cal_loss' in conv_turn:
                cal_loss = conv_turn['cal_loss']

            message = ''
            message_role = ''
            n_segs = len(conv_turn['content_list'])
            for i, conv_seg in enumerate(conv_turn['content_list']):
                content: str = conv_seg['content']
                if i == 0:
                    content = content.lstrip()
                if i == n_segs - 1:
                    content = content.rstrip()

                content_type = conv_seg['type']
                if content_type == 'text':
                    message += content
        
                    message_role += content
                elif content_type == 'audio':
                    audio_token_str = self.convert_audioID2str(conv_seg['audio_infos']['audio_tokens'])
                    conv_seg['audio_infos']['audio_tokens'] = []
                    audio_str = f'<|beginofaudio|>{audio_token_str}<|endofaudio|>'

                    if role_type != 'assistant':
                        message_role += audio_token_str

                    if '<|audio|>' in content:
                        content = content.replace('<|audio|>', audio_str)
                        message += content
                    else:
                        print(content)
                        message += audio_str
                elif content_type == 'audio_text':
                    audio_token_str = self.convert_audioID2str(conv_seg['audio_infos']['audio_tokens'])
                    conv_seg['audio_infos']['audio_tokens'] = []
                    audio_str = f'<|inter_audio_begin|>{audio_token_str}<|inter_audio_end|>'
                    
                    if '<|audio|>' in content:
                        content = content.replace('<|audio|>', audio_str)
                    else:
                        raise ValueError()
                    message += content

                    if role_type == 'assistant':
                        message_role += conv_seg['audio_infos']['text']
                    else:
                        message_role += content
                elif content_type == 'cot':
                    cot_str = f'<|cot_begin|>{content}<|cot_end|>\n'
                    message += cot_str
                    message_role += cot_str
                elif 'tool' in content_type:
                    tool_str = f'<tool_call>{content}</tool_call>\n'
                    message += tool_str
                    message_role += tool_str
                else:
                    raise ValueError(f'{content_type} not supported yet~')
                
            # 完整的信息 
            turn_in_str = self.conv_template.format(
                role_name=role_name, 
                message=message)
            tokens_t = self.tokenizer.tokenize(turn_in_str.strip())

            # 为了归一化sft的answer长度，因此loss mask要除以answer的长度
            tokens_len = len(tokens_t[:-1])

            if cal_loss:
                if len(dialoge) <= 2 or (index + 1) % 4 == 0:
                    new_sample = {
                        "model_input": sample_in_str + '<|im_start|>assistant\n',
                        "better_context": turn_in_str.replace('<|im_start|>assistant\n', '').strip(),
                        "better_context_len": tokens_len,
                        "candidate_set": [],
                        "task_category": task
                    }
                    samples.append(new_sample)
            
            sample_in_str += turn_in_str.strip()

        return samples
    

def extend_to_n_candidates(org_datas: list, n_candidate):
    prompts = []
    for ele in org_datas:
        prompts.extend([ele['model_input']] * n_candidate)

    return prompts


def run_vllminference_one_gpu(args, proc_id, new_datas, model_name, sampling_params, GPUS_PER_PROC):
    devices  = ""
    for i in range(GPUS_PER_PROC):
        if i >0:
            devices += ',' + str(GPUS_PER_PROC*proc_id+i)
        else:
            devices += str(GPUS_PER_PROC*proc_id+i)
    print(devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    llm = LLM(model=model_name, dtype='auto')
    print(f'Data length: {len(new_datas)}')

    chunk_size = args.save_chunk_size
    chunks = len(new_datas)//chunk_size
    
    for i in range(chunks+1):
        if i == chunks and len(new_datas) % chunk_size != 0:
            sub_prompt_list = new_datas[chunks*chunk_size: ]
        elif i < chunks:
            sub_prompt_list = new_datas[i*chunk_size: (i+1)*chunk_size]
        else:
            continue
        prompts = extend_to_n_candidates(sub_prompt_list, args.candidate_num)
        sub_result  = llm.generate(prompts, sampling_params)
        print(f'VLLM results: {len(sub_result)}')
        output_file = f'{args.output_dir}/output_process_g{args.group_num}_{proc_id}_{i}.json'
        write_output(args, sub_result, sub_prompt_list, output_file)


def build_actor_context(system_prompt, convs):
    pmt = system_prompt + '\n'
    history = convs
    if len(history) > 0:
        for line in history:
            pmt += f"<{line['name']}>[{line['date']}]:{line['context'] if line['context'] else line['better_context']} </s>\n"
    return pmt


def load_data(args, encoder: Encoder):
    """
    1. 加载数据，并实现单条样本仅最后一句计算loss
    2. 构造模型请求的输入，方便后续分布式请求

    """

    datas = []
    input_dir = args.input_dir
    with open(input_dir, 'r') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in tqdm(lines, desc="Loading Data"):
            samples = encoder.encode(line)
            if samples is not None and len(samples) > 0:
                datas.extend(samples)
            if len(datas) >= 500000:
                break
    print(f'Org data length: {len(datas)}')
    for sample in datas[:100]:
        print(f'{sample["model_input"] + sample["better_context"]}')
        print("*" * 15)

    return datas 


def load_data_test():
    args = get_args()
    load_data(args=args)
    

def filter_multi_turn():
    samples = []
    for i in range(4):
        data_path = f'/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/20250606/org_data/lpo_train_{i + 1}.jsonl'
        with open(data_path, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)
            for line in tqdm(lines, desc="Loading Data"):
                item = json.loads(line)
                if 'dialog' in item:
                    dialoge = item['dialog']
                else:
                    dialoge = item['dialoge']

                # 增加role dialog膨胀的对话数据集
                dialoge = json.loads(dialoge) if isinstance(dialoge, str) else dialoge
                if len(dialoge) > 2:
                    samples.append(item)
    save_path = '/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/20250611/org_data/lpo_train_0.jsonl'
    with open(save_path, 'w') as f:
        for s in samples:
            line = json.dumps(s, ensure_ascii=False)
            f.write(line)
            f.write('\n')


def main(args):
    # 分布式初始化
    torch.multiprocessing.set_start_method('spawn')
    gpus_per_proc = args.gpus_per_proc
    print('torch.cuda.device_count:', torch.cuda.device_count())
    NUM_PROCS = torch.cuda.device_count() // gpus_per_proc
    print(f'gpus_per_proc :{gpus_per_proc}, NUM_PROCS: {NUM_PROCS}')

    encoder = Encoder(args=args)

    # 数据加载
    new_datas = load_data(args, encoder)
    raw_num_data = len(new_datas)
    split_prompts = []
    print(f'num of datas {len(new_datas)}')

    # 数据拆分
    num_data_per_procs = raw_num_data // NUM_PROCS if raw_num_data%NUM_PROCS==0 else (raw_num_data // NUM_PROCS + 1)
    print(f'num of num_data_per_procs {num_data_per_procs}')
    last_idx = 0
    for i in range(NUM_PROCS - 1):
        split_prompts.append(new_datas[i * num_data_per_procs: (i + 1) * num_data_per_procs])
        last_idx = (i + 1) * num_data_per_procs
    split_prompts.append(new_datas[last_idx: ])
    print(len(split_prompts[0]))
    if not os.path.exists(args.output_dir): 
        os.makedirs(args.output_dir)
    # 推理
    stop_tokens = ["<|endoftext|>"]
    stop_ids = encoder.tokenizer.tokenizer.convert_tokens_to_ids(stop_tokens)
    sampling_params = SamplingParams(temperature=args.temp, top_p=args.topp, stop_token_ids=stop_ids, max_tokens=8192)
    inputs = [(args, i, part, args.model_path, sampling_params, gpus_per_proc) 
              for i, part in enumerate(split_prompts)]

    with multiprocessing.Pool(processes=NUM_PROCS) as pool:
        pool.starmap(run_vllminference_one_gpu, inputs)


if __name__ == '__main__':
    start = time.time()
    args = get_args()
    if not args.skip:
        main(args=args)
    else:
        print('Skip candidate generation.')
    print(f'total time is {time.time()-start}')
