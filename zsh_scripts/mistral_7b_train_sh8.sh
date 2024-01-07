#!/bin/bash

export http_proxy=100.66.28.72:3128
export https_proxy=100.66.28.72:3128
export HTTP_PROXY=100.66.28.72:3128
export HTTPS_PROXY=100.66.28.72:3128

# export http_proxy=100.66.27.151:3128
# export https_proxy=100.66.27.151:3128
# export HTTP_PROXY=100.66.27.151:3128
# export HTTPS_PROXY=100.66.27.151:3128

export TMPDIR=/ML-A800/home/xiangyue/yfli/tmp
export HF_HOME=/ML-A800/home/xiangyue/yfli/.cache/huggingface


export MODEL_DIR=/ML-A800/models/Mistral-7B-v0.1
export WANDB_ENTITY=flyhero99
export WANDB_PROJECT=attribution-eval-v3.0

# pip install jsonlines
# pip install backoff
# pip install anthropic

# 32 1e-5
# ***************** Set parameters here *****************
dataset_version=v3.0
model=Mistral-7B-v0.1
template=base_c_e
lr=1e-5
num_train_epoches=2
start_gpu_index=4
master_port=34567  # port
per_device_train_batch_size=1
gas=8
nodes=4
# ***************** The followings are auto-calculated parameters *****************
cuda_devices=$(seq -s ',' $start_gpu_index $(($start_gpu_index + $nodes - 1)))
export CUDA_VISIBLE_DEVICES=$cuda_devices
bs=$((gas * nodes))
eval_bs=$((per_device_train_batch_size * 2))
setting=template-${template}-bs${bs}-lr${lr}-gas${gas}
current_time=$(date +"%Y-%m-%d-%H:%M:%S")

echo ${CUDA_VISIBLE_DEVICES}
# make sure you want to do the deletion
# ************************************************************************************
export OUTPUT_DIR=../checkpoints/${model}-${dataset_version}-${setting}

rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}
torchrun --nproc_per_node ${nodes} --master-port ${master_port} ../src/train/run_mistral_7b.py \
  --model_name_or_path $MODEL_DIR \
  --template ${template} \
  --template_path ../src/prompts.json \
  --dataset_version ${dataset_version} \
  --data_path ../data_1216/AttributionBench \
  --num_train_samples -1 \
  --bf16 True \
  --output_dir $OUTPUT_DIR \
  --model_max_length 1024 \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size ${eval_bs} \
  --gradient_accumulation_steps ${gas} \
  --num_train_epochs ${num_train_epoches} \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --save_strategy epoch \
  --logging_strategy steps \
  --logging_steps 10 \
  --learning_rate ${lr} \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --fsdp 'full_shard auto_wrap' \
  --fsdp_transformer_layer_cls_to_wrap 'MistralDecoderLayer' \
  --tf32 True \
  --report_to wandb
