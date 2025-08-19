import json
import os
import traceback

from tqdm import tqdm
import numpy as np


def read_data(data_path='/cognitive_comp/wangrui/data/1207/calc.json', rank=0):
    with open(data_path) as f:
        new_data = []
        for line in tqdm(f, desc=f'Rank{rank} Loading {data_path.split("/")[-1]}'):
            try:
                line_json = json.loads(line)
                new_data.append(line_json)
            except Exception as e:
                print(data_path)
                print(line)
                traceback.print_exc()
            
    return new_data


def read_json(data_path):
    try:
        with open(data_path) as f:
            line_json = json.load(f)
    except Exception as e:
        print(data_path)
        traceback.print_exc()
            
    return line_json


def write_to_json(data,
                  data_path='/cognitive_comp/wangrui/data/1207/math/train.json', 
                  is_beautify=False):
    print(f'Path: {data_path}, Nsamples: {len(data)}')
    with open(data_path, 'w') as f:
        for line in tqdm(data):
            if not is_beautify:
                f.write(json.dumps(line, ensure_ascii=False))
            else:
                f.write(json.dumps(line, ensure_ascii=False, indent=4))
            f.write('\n')
            
            
def merge_jsonl_files(file_path, merged_name):
    merged_file_path = os.path.join(file_path, merged_name)
    # 获取文件夹中所有的jsonl文件
    if not os.path.exists(merged_file_path):
        file_list = [file for file in os.listdir(file_path) if file.endswith('.json') and file.startswith('output')]
        print(f"All files: {file_list}")
        with open(merged_file_path, 'w') as outfile:
            for file_name in file_list:
                print(file_name)
                with open(os.path.join(file_path, file_name), 'r') as infile:
                    for line in tqdm(infile):
                        outfile.write(line)
                        
                        
def plot_attention_matrix(mat, labels, fig_num):                     
    import matplotlib.pyplot as plt

    fig_path = f'log/attention_mask_{fig_num}.png'
    plt.matshow(mat, cmap=plt.cm.gray)
    plt.savefig(fig_path)
    plt.close()


if __name__ == '__main__':
    merge_jsonl_files(file_path='/cognitive_comp/wangrui/data/0328/source_split_rag_noise',
                      merged_name='rag_noise_xpo.json')