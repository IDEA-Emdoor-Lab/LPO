# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import math
import json
import os
import sys
import re
from tqdm import tqdm
import random
import multiprocessing as mp

import numpy as np

from utils.ddp_tools import print_rank_0
from utils.common_utils import read_data


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import multiprocessing

from megatron.core.datasets import indexed_dataset


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=False, default='GPT2BPETokenizer',
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer', 'Llama2Tokenizer',
                                'NullTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--sequence-packing',action='store_true', help='packing sequence')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='YTTM tokenizer model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--debug', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    group.add_argument(
        '--patch-tokenizer-type',
        type=str,
        required=True,
        choices=['Qwen2Tokenizer', 'QWen', 'LLamaTokenizer', 'DeepSeekV2Tokenizer', 'LLama3Tokenizer'],
        help='What type of tokenizer to use.',
    )
    group.add_argument('--load',
                       type=str,
                       default=None,
                       help='path to tokenizer config file')

    group.add_argument('--seq-length',
                       type=int,
                       default=2048,
                       help='sequence length')
    
    group.add_argument('--threshold',
                       type=float,
                       default=1.0,
                       help='filtering prob')

    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=0,
                       help='extra_vocab_size')
    group.add_argument(
        "--split_size", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--n_candidate", type=int, default=-1)

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")
    
    if args.sequence_packing:
        print('Use internal single-threaded sequence packing..')
    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True

def print_processing_stats(count, proc_start, total_bytes_processed, args):
    if count % args.log_interval == 0:
        current = time.time()
        elapsed = current - proc_start
        mbs = total_bytes_processed/elapsed/1024/1024
        print(f"Processed {count} documents",
                f"({count/elapsed} docs/s, {mbs} MB/s).",
                file=sys.stderr)

def process_doc(doc, encoder):
    """
    处理单个文档的函数，调用 encoder.encode 并返回结果列表
    """
    return list(encoder.encode(doc))


def encode_lpo_data_to_one_line(sample: dict, seq_len):
    task = sample['task']
    if task == 'common_voice_tts':
        print(task)
        return None

    chosen_sample = sample['chosen']
    chosen_input_ids = chosen_sample['input_ids']
    chosen_loss_mask_idx = chosen_sample['loss_mask_idx']
    chosen_logprob = chosen_sample['logprob']
    chosen_seq = chosen_input_ids + chosen_loss_mask_idx + [chosen_logprob]
    answer_len = chosen_loss_mask_idx[1] - chosen_loss_mask_idx[0]
    if answer_len <= 0 or chosen_loss_mask_idx[0] >= seq_len or chosen_loss_mask_idx[1] >= seq_len:
        print(f'Illegal chosen begin end index pair: {chosen_loss_mask_idx}', flush=True)
        return None

    reject_sample = sample['reject']
    reject_input_ids = reject_sample['input_ids']
    reject_loss_mask_idx = reject_sample['loss_mask_idx']
    reject_logprob = reject_sample['logprob']
    reject_seq = reject_input_ids + reject_loss_mask_idx + [reject_logprob]
    answer_len = reject_loss_mask_idx[1] - reject_loss_mask_idx[0]
    if answer_len <= 0 or reject_loss_mask_idx[0] >= seq_len:
       print(f'Illegal reject begin end index pair: {reject_loss_mask_idx}', flush=True)
       return None
    
    if chosen_logprob - reject_logprob > 2.0 and reject_loss_mask_idx[1] < seq_len:
        print(f'Chosen logprob {chosen_logprob} is much larger than Reject logprob {reject_logprob}')
        return None

    one_line_sample = chosen_seq + reject_seq
    one_line_sample_float = [float(ele) for ele in one_line_sample]

    return one_line_sample_float


def process_file(args):
    rank, file_path, seq_len, threshold = args
    print(args)
    # 处理单个文件的函数
    data_list = read_data(data_path=file_path, rank=rank)

    encoded_docs = []
    for sample in tqdm(data_list, desc=f'Rank{rank} Converting to one line'):
        ids = {}
        lens = {}
        is_dropped = random.random()
        if is_dropped > threshold:
            continue
        tokens = encode_lpo_data_to_one_line(sample, seq_len)
        if tokens is None:
            print('Sample exceed seq length, dropped.', flush=True)
            continue
        assert len(tokens) == (seq_len + 3) * 2
        ids['text'] = tokens
        lens['text'] = [len(tokens)]
        encoded_docs.append([ids, lens, len(json.dumps(ids))])

    return encoded_docs


def main():
    args = get_args()
    print(f'args is:{args}', flush=True)

    assert args.workers % args.partitions == 0
    
    startup_start = time.time()
    if not args.sequence_packing:
        return
     
    print(f'input data path is:{args.input}', flush=True)
    all_pathes = [os.path.join(args.input, name) for name in os.listdir(args.input) 
                  if not os.path.isdir(os.path.join(args.input, name))]
    # 创建进程池
    n_threads = 32
    all_data = []
    with mp.Pool(processes=n_threads) as pool:
        # 使用进程池并行处理文件
        results = pool.map(process_file, [(i, p, args.seq_length, args.threshold) 
                                            for i, p in enumerate(all_pathes)])
    # 将所有数据合并
    for data_list in tqdm(results, desc='Merge data'):
        print(f'Data length: {len(data_list)}')
        all_data.extend(data_list)
    if args.n_candidate <= 1:
        print('Shuffling all data...')
        random.shuffle(all_data)
        print('Shuffle data done.')
    elif args.n_candidate > 1:
        print('顺序执行epoch数据')
        new_all_data = []
        for i in range(args.n_candidate):
            data_t = all_data[i::args.n_candidate]
            random.shuffle(data_t)
            new_all_data.extend(data_t)
        all_data = new_all_data
    
    encoded_docs = all_data
    level = "document"
    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    for key in args.json_keys:
        output_bin_files[key] = "{}/{}_{}.bin".format(args.output_prefix, key, level)
        output_idx_files[key] = "{}/{}_{}.idx".format(args.output_prefix, key, level)
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=np.float32,
        )

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)
    cnt = 1
    for datas in encoded_docs:
        #for (doc, sentence_lens, bytes_processed) in datas:
        doc, sentence_lens, bytes_processed = datas
        total_bytes_processed += bytes_processed
        for key in doc.keys():
            builders[key].add_document(doc[key], sentence_lens[key])
        print_processing_stats(cnt, proc_start, total_bytes_processed, args)
        cnt += 1
    print(f"After pre-tokenizing, the idxmap dataset has {cnt - 1} samples")

    builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':
    main()