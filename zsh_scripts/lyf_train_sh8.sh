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


# export MODEL_DIR=/ML-A800/models/Llama-2-7b-chat-hf
export MODEL_DIR=/ML-A800/models/Llama-2-7b-hf
export WANDB_ENTITY=flyhero99
export WANDB_PROJECT=attribution-eval-v3.0

dataset_versions=("v3.0" "v3.1" "v3.2" "v3.3")
templates=("base_c_e" "base_c_e_r" "base_q_c_e" "base_q_c_e_r")

for dataset_version in "${dataset_versions[@]}"; do
  for template in "${templates[@]}"; do
    model=$(basename $MODEL_DIR)
    num_train_epoches=2
    start_gpu_index=0
    master_port=$((RANDOM + 10000))
    per_device_train_batch_size=1
    lr=1e-5
    gas=8
    nodes=4
    cuda_devices=$(seq -s ',' $start_gpu_index $(($start_gpu_index + $nodes - 1)))
    export CUDA_VISIBLE_DEVICES=$cuda_devices
    bs=$((gas * nodes))
    eval_bs=$((per_device_train_batch_size * 2))
    setting=template-${template}-bs${bs}-lr${lr}-gas${gas}
    current_time=$(date +"%Y-%m-%d-%H:%M:%S")
    export OUTPUT_DIR=../checkpoints/${model}-${dataset_version}-${setting}
    rm -rf $OUTPUT_DIR
    WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}
    
    # 运行实验
    torchrun --nproc_per_node ${nodes} --master-port ${master_port} ../src/train/run_train_with_question.py \
      --model_name_or_path $MODEL_DIR \
      --template ${template} \
      --template_path ../src/prompts.json \
      --dataset_version ${dataset_version} \
      --data_path ../data_1216/AttributionBench \
      --num_train_samples -1 \
      --bf16 True \
      --output_dir $OUTPUT_DIR \
      --model_max_length 2048 \
      --per_device_train_batch_size ${per_device_train_batch_size} \
      --per_device_eval_batch_size ${eval_bs} \
      --gradient_accumulation_steps ${gas} \
      --num_train_epochs ${num_train_epoches} \
      --evaluation_strategy steps \
      --eval_steps 100 \
      --evaluation_strategy no \
      --save_strategy no \
      --save_total_limit 1 \
      --logging_strategy steps \
      --logging_steps 10 \
      --learning_rate ${lr} \
      --weight_decay 0. \
      --warmup_ratio 0.03 \
      --lr_scheduler_type cosine \
      --fsdp 'full_shard auto_wrap' \
      --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
      --tf32 True \
      --report_to wandb
  done
done