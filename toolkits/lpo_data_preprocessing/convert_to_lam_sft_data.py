# Copyright (c) 2021, EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Processing data for pretraining."""

import argparse
import os
import json
import glob

import random
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

LOG_DATA_NUM = 1


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

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
            "HFGPTNeoXTokenizerFast",
            "SPMTokenizer",
            "LlamaTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )

    group.add_argument("--ftfy", action="store_true", help="Use ftfy to clean text")
    group = parser.add_argument_group(title="output data")
    group.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to binary output file without suffix",
    )

    group = parser.add_argument_group(title="runtime")
    group.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--split_size", type=int, default=1, help="Number of worker processes to launch"
    )
    group.add_argument(
        "--epoch",
        type=int,
        default=1,
    )
    group.add_argument(
        "--seq_length",
        type=int,
        default=512,
    )
    group.add_argument(
        "--log-data",
        action="store_true",
        default=False,
        help="whether log data",
    )

    group.add_argument(
        "--concat",
        action="store_true",
        default=False,
        help="whether to concat data",
    )
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def save_data(data, file_path):
    sv_dir = "/".join(file_path.split("/")[:-1]) + "/"
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    with open(file_path, "w", encoding="utf8") as f:
        for line in tqdm(data, desc=f'{file_path.split("/")[-1]}'):
            json_data = json.dumps(line, ensure_ascii=False)
            f.write(json_data + "\n")


def read_parquet(parquet_path: str, dir_name: str) -> pd.DataFrame:
    file_paths = parquet_path
    # 读取所有Parquet文件并将它们合并成一个数据框
    df_list = []
    
    for file in tqdm(file_paths, desc=dir_name):
        df_list.append(pd.read_parquet(file))
    df = pd.concat(df_list, ignore_index=True).to_dict(orient='records')

    return df


def find_parquet_files(directory):
    parquet_files = []
    
    # 使用glob递归查找所有.parquet文件
    for file_path in glob.glob(os.path.join(directory[0], '**', '*.parquet'), recursive=True):
        parquet_files.append(file_path)
    
    return parquet_files


def get_folders(input_path):
    # 获取指定路径下的所有文件和文件夹
    items = os.listdir(input_path)
    
    # 过滤出文件夹
    folders = [(os.path.join(input_path, item), item) for item in items if os.path.isdir(os.path.join(input_path, item))]
    print(f'Folders:\n{folders}')
    parquets_list = {}
    for folder in folders:
        parquets_path = find_parquet_files(directory=folder)
        parquets_list[folder[1]] = parquets_path
    
    return parquets_list


task_ratio = {
    'audio_desp': 0.5,
    # 'role_dialog': 1.0,
    'emo_dection': 1.0,
    'emo_desp_clf': 0.1,
    'multi_AUDIO_ASR': 0.1,
    'speaker_recognition': 0.05,
    'zeroshot_tts': 1.0,
    'text_sft': 0.2
}

task_ratio2 = {
    'zeroshot_tts': 1.0
}

task_ratio3 = {
    'emilia_tts_xpo_0326': 1.0
}

task_ratio4 = {
    'emilia_tts_xpo_mixed_0422': 1.0
}


def split_alm_sft_data(input_path):
    parquets_dir = get_folders(input_path)
    name2num = {}
    sft_data_list = []
    lpo_data_list = []
    for file_name, parquet_list in parquets_dir.items():
        if file_name not in task_ratio4:
            print(file_name)
            continue
        dataset = read_parquet(parquet_list, file_name)
        ratio = task_ratio4[file_name]
        name2num[file_name] = 0
        nums = 0
        for row in tqdm(dataset, desc=file_name):
            if random.random() < ratio:
                row['task_category'] = file_name
                row['task'] = file_name
                row['dialog'] = json.loads(row['dialog'])
                nums += 1
                if random.random() < 0.0:
                    sft_data_list.append(row)
                else:
                    lpo_data_list.append(row)
        name2num[file_name] = nums
    print(json.dumps(name2num, indent=4, ensure_ascii=False))
    date='20250423'
    save_data(data=sft_data_list, 
              file_path=f'/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/{date}/org_data/sft_train.jsonl')
    n_samples = 180000
    splitted_data = [lpo_data_list[i:i + n_samples] 
                     for i in range(0, len(lpo_data_list), n_samples)]
    for i, data in enumerate(splitted_data):
        save_data(data=data, 
                  file_path=f'/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/{date}/org_data/lpo_train_{i}.jsonl')


if __name__ == "__main__":
    split_alm_sft_data(input_path='/cognitive_comp/ccnl_common_data/large/large_audio_model_text/audio/output/lam/instruct_data')