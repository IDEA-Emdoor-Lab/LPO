#! /bin/bash
START_TIME=$SECONDS

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
MEGATRON_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATH}:${MEGATRON_PATH}/PAI-Megatron-LM-240718

date=20250214
DATA_PATH=/cognitive_comp/ccnl_common_data/wangrui/alm_sft_training
output_file=${DATA_PATH}/${date}/train_data/mmap_data
mkdir -p ${output_file}
echo ${output_file}

input_data_path=${DATA_PATH}/${date}/train_data
tokenizer=Qwen2Tokenizer
seq_len=4096
output_data_path=${output_file}
step=8600
ref_model_path=/cognitive_comp/ccnl_common_data/large/checkpoints/sft/20240208/checkpoint/finetune-mcore-qwen2.5-7B-lr-9e-6-minlr-3e-6-bs-1-gbs-128-seqlen-16384-pr-bf16-tp-2-pp-4-cp-1-ac-false-do-true-sp-true-ti-10000-wi-10/iter_000${step}_hf
default_packing=true

if [ -z ${default_packing} ]; then
  default_packing=false
fi

if [ $default_packing = true ]; then
  packing_option="\
    --sequence-packing 
  "
else
  packing_option=""
fi

cmd="python build_idxmap_sft_dataset.py \
  --input ${input_data_path} \
  --output-prefix ${output_data_path} \
  --patch-tokenizer-type ${tokenizer} \
  --load ${ref_model_path} \
  --seq-length ${seq_len} \
  --workers 1 \
  --partitions 1 ${packing_option} \
  --split_size 9000 "

echo $cmd
eval $cmd

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "$(($ELAPSED_TIME/60)) min $(($ELAPSED_TIME%60)) sec"
