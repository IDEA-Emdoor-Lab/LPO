#!/bin/bash
#SBATCH --job-name=lam_lpo # create a short name for your job
#SBATCH -N 1 # node count
#SBATCH --gres=gpu:hgx:8 # number of gpus per node
#SBATCH --ntasks-per-node 1 # number of tasks to run per node
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=24G
#SBATCH -p pos # -preempted    #SBATCH --exclude ccnl08,ccnl07,ccnl09,ccnl10  -preempted
#SBATCH -o log/%x-%j.log # output and error log file names (%x for job id)

num_gpus=8
date=20250228
step=0
CODE_PATH=./toolkits/lpo_data_preprocessing  # code path 当前代码路径
DATA_PATH=data_path
candidate_num=3 # 生成的candidate数据量
micro_batch_size=$candidate_num

ref_model_path=model_path # 生成样本的candidate path


seq_length=4097
tokenizer_type=QWen

##########################step1 candidate#############################
echo "step1 candidate generation begin."
tokenizer_path=${ref_model_path}
echo ${tokenizer_path}

dir=${DATA_PATH}/${date}
mkdir -p ${dir}
echo ${dir}

input_dir=/cognitive_comp/ccnl_common_data/alm_sft_training/${date}/org_data/lpo_train.jsonl
echo ${input_dir}

candidate_dir=${dir}/candidate/
mkdir -p ${candidate_dir}
echo ${candidate_dir}

# vllm生成候选回答。
timer_start=`date "+%Y-%m-%d %H:%M:%S"`

python -u $CODE_PATH/gene_lam_xpo_candidate.py \
        --input_dir  ${input_dir} \
        --model_path ${ref_model_path} \
        --tokenizer_path ${tokenizer_path} \
        --output_dir ${candidate_dir} \
        --candidate_num ${candidate_num} \
        --gpus_per_proc 1 \
        --num_gpus ${num_gpus} \
        --save_chunk_size 10000 \
        # --skip

timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "开始： $timer_start"
echo "结束： $timer_end"
echo "耗时： $duration"
echo "step1 candidate generation finished."

##########################step2 logits#############################
echo "step2 logits generation begin."
nnodes=1
gpus_per_node=$num_gpus

codepath=$CODE_PATH/gene_lam_xpo_logits.py
echo ${codepath}

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
master_port=$(shuf -n 1 -i 40000-65535)
echo Node IP: $head_node_ip
export LOGLEVEL=INFO

input_file=${candidate_dir}
echo ${input_file}

output_file=${DATA_PATH}/${date}/train
mkdir -p ${output_file}
echo ${output_file}

tokenizer_path=${ref_model_path}
echo ${tokenizer_path}

timer_start=`date "+%Y-%m-%d %H:%M:%S"`

srun torchrun \
    --nnodes=$nnodes \
    --nproc_per_node=$gpus_per_node \
    --max_restarts=1 --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:$master_port \
    $codepath \
    --input  ${input_file} \
    --output ${output_file} \
    --ref_model_path ${ref_model_path} \
    --vocab-file ${tokenizer_path} \
    --tokenizer-type ${tokenizer_type} \
    --epoch 1 \
    --seq_length ${seq_length} \
    --save_chunk_size 3000 \
    --micro_batch_size ${micro_batch_size} \
    --loss_type sigmoid \
    --average_logprobs \
    --skip

timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "开始： $timer_start"
echo "结束： $timer_end"
echo "耗时： $duration"
echo "step2 logits generation finished."

##########################step3 Convert to MMap#############################
echo "step3 Converting to mmap begin."

MEGATRON_PATH=../../pai-megatron-patch/PAI-Megatron-LM-240718
MEGATRON_PATCH_PATH=../../pai-megatron-patch/megatron_patch
export PYTHONPATH=$PYTHONPATH:${MEGATRON_PATCH_PATH}:${MEGATRON_PATH}

input_data_path=${output_file}/jsonl_data
mmap_data_path=${output_file}/mmap_data
mkdir -p ${mmap_data_path}
echo ${mmap_data_path}

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

codepath=$CODE_PATH/build_idxmap_lpo_dataset.py
echo ${codepath}

timer_start=`date "+%Y-%m-%d %H:%M:%S"`

python ${codepath} \
  --input ${input_data_path} \
  --output-prefix ${mmap_data_path} \
  --patch-tokenizer-type ${tokenizer_type} \
  --load ${ref_model_path} \
  --seq-length ${seq_length} \
  --workers 1 \
  --partitions 1 ${packing_option} \
  --split_size 9000

timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "开始： $timer_start"
echo "结束： $timer_end"
echo "耗时： $duration"
echo "step3 Converting to mmap end."