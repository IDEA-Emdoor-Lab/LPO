import os
import torch
import torch.distributed as dist
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import json
from typing import List
import torch.nn.functional as F

def setup():
    # 初始化分布式多进程
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group('nccl')

def cleanup():
    # 关闭所有进程
    dist.destroy_process_group()

def load_data(file_path, is_training=False):
    # 加载sft类似格式的数据
    with open(file_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        result=[]
        for line in lines: 
            data = json.loads(line)
            # if data['input']!=None and data['output']!=None and data['input']!='' and data['output']!='':
            result.append(data)
        return result

def save_data(data,file_path):
    # 保存dict列表数据到json文件
    final_dir = os.path.dirname(file_path)
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    print(f"lines of file {file_path} is {len(data)}.")
    with open(file_path, 'w', encoding='utf8') as f:
        for line in data:
            json_data=json.dumps(line,ensure_ascii=False)
            f.write(json_data+'\n')

def load_model(ckpt, device_id):
    # 分布式加载模型
    print(f'model name is {ckpt}')
    model = AutoModelForCausalLM.from_pretrained(ckpt).half()
    model = nn.parallel.DistributedDataParallel(model.to(device_id),
                                                device_ids=[device_id],
                                                output_device=device_id)
    model.eval()
    print('after load model:')
    print(os.popen('nvidia-smi').read())
    return model

def load_tokenizer(ckpt):
    # 加载tokenizer,以下特殊token是llama模型的
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    
    return tokenizer

def sort_list_of_dicts(lst, key):
    # 基于key对dict列表进行排序，主要用于ppl排序
    return sorted(lst, key=lambda x: x[key])

def xpo_collator(batch):
    # collator_fn for dataloader
    
    return batch

def dynamis_loader(sequence_len_of_batch_size, max_tokens_num, real_batch, batch, keep_loader):
    # 根据当前的最大token_num以及目前的batchsize，判断是否要继续loader，最大化batch size
    for seq_len in sequence_len_of_batch_size:
        if max_tokens_num > (seq_len) and max_tokens_num < (seq_len*2):
            if seq_len*2 not in sequence_len_of_batch_size: 
                sequence_len_of_batch_size[seq_len*2] = 1
            if len(real_batch['tokens_num']) < sequence_len_of_batch_size[seq_len*2]:
                keep_loader = True
            else:
                keep_loader = False
            break
        elif max_tokens_num == (seq_len) or max_tokens_num < (seq_len):
            if len(real_batch['tokens_num']) < sequence_len_of_batch_size[seq_len]:
                keep_loader = True
            else:
                keep_loader = False
            break
    return keep_loader, real_batch


def gather_all_xpo_data(local_rank, res_datas, world_size):
    # 聚合所有进程的object到一起s
    # 不加这句话会报错，原因可能是：当后端为nccl时，运行dist.gather_object()要求当前进程独占GPU。
    # 使用gloo后端也可以避免报错。
    torch.cuda.set_device(local_rank)
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, res_datas)
    res = []
    for o in output:
        res.extend(o)
    return res


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)