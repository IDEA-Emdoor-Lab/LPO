#!/bin/bash
#SBATCH --job-name=mindbot # create a short name for your job
#SBATCH --nodes=1 # node count
#SBATCH --ntasks-per-node=1 # total number of tasks across all nodes
#SBATCH --cpus-per-task=16 # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:hgx:2 # number of gpus per node
#SBATCH -w hgx029 # number of gpus per node
#SBATCH -p pos # number of gpus per node

#SBATCH -o ./log/server.log # output and error log file names (%x for job id)

audio_model_path=/cognitive_comp/common_checkpoint/S_model_management/codec/20241017/Qwen2.5-7B-Codec0927-S204000-AEdivided50/g_00204000
step=48000
model_path=/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/20250616/train/checkpoint/xpo-mcore-qwen2.5-7B-lr-2e-7-minlr-1e-7-bs-6-gbs-24-seqlen-4096-pr-bf16-tp-2-pp-4-cp-1-ac-false-do-true-sp-true-ti-35000-wi-66/iter_000${step}_hf #/cognitive_comp/ccnl_common_data/large/checkpoints/sft/20240214/checkpoint/finetune-mcore-qwen2.5-7B-lr-9e-6-minlr-3e-6-bs-1-gbs-128-seqlen-16384-pr-bf16-tp-2-pp-4-cp-1-ac-false-do-true-sp-true-ti-10000-wi-10/iter_0007200_hf #/cognitive_comp/ccnl_common_data/large/checkpoints/sft/20240208/checkpoint/finetune-mcore-qwen2.5-7B-lr-9e-6-minlr-3e-6-bs-1-gbs-128-seqlen-16384-pr-bf16-tp-2-pp-4-cp-1-ac-false-do-true-sp-true-ti-10000-wi-10/iter_0009000_hf #/cognitive_comp/ccnl_common_data/large/checkpoints/sft/20240208/checkpoint/finetune-mcore-qwen2.5-7B-lr-9e-6-minlr-3e-6-bs-1-gbs-128-seqlen-16384-pr-bf16-tp-2-pp-4-cp-1-ac-false-do-true-sp-true-ti-10000-wi-10/iter_0008600_hf #/cognitive_comp/ccnl_common_data/large/checkpoints/sft/20240208/checkpoint/finetune-mcore-qwen2.5-7B-lr-9e-6-minlr-3e-6-bs-1-gbs-128-seqlen-16384-pr-bf16-tp-2-pp-4-cp-1-ac-false-do-true-sp-true-ti-10000-wi-10/iter_0007500_hf #/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training/20250206/checkpoint/finetune-mcore-qwen2.5-7B-lr-9e-6-minlr-5e-6-bs-1-gbs-48-seqlen-16384-pr-bf16-tp-2-pp-4-cp-1-ac-false-do-true-sp-true-ti-10000-wi-10/iter_0008400_hf

model_config=/cognitive_comp/common_checkpoint/S_model_management/codec/20241017/Qwen2.5-7B-Codec0927-S204000-AEdivided50/codec_config.json #"/cognitive_comp/common_checkpoint/S_model_management/codec/20240923/0923_24k_3s/model_config.json"
ckpt_config=/cognitive_comp/common_checkpoint/S_model_management/codec/20241017/Qwen2.5-7B-Codec0927-S204000-AEdivided50/

use_vllm=true

export GRADIO_TEMP_DIR=/cognitive_comp/wangrui/codes/pai-megatron-patch/toolkits/webdemo/log/tmp
CUDA_VISIBLE_DEVICES=0,1 python src/webdemo_audio.py \
    --host 0.0.0.0 \
    --port 8890 \
    --model_config $model_config \
    --ckpt_config $ckpt_config \
    --model_path $model_path \
    --demo_url /demo/541832 \
    --api_url /api/demo/541832 \
    --use_vllm $use_vllm

#--use_vllm $use_vllm
# --demo_url /demo/541832 \
#--api_url /api/demo/541832 \

