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

cd /ML-A800/home/xiangyue/yfli/AttributionBench/zsh_scripts

# export MODEL_DIR=/ML-A800/models/flan-t5-large
# export MODEL_DIR=/ML-A800/home/xiangyue/yfli/hf_models/ul2
export WANDB_ENTITY=flyhero99
export WANDB_PROJECT=attribution-eval-v3.0-newdata

pip install jsonlines
pip install backoff
pip install anthropic

# zero-shot
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method autoais \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/t5_xxl_true_nli_mixture \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# zero-shot
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method attrbench \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/Llama2-7b-hf \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# zero-shot
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method autoais \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/flan-t5-xxl \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# zero-shot
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method autoais \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/ul2-flan \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# zero-shot
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method autoais \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/flan-t5-xl \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# zero-shot
export CUDA_VISIBLE_DEVICES="0"
python ../src/inference/run_inference.py \
    --method autoais \
    --data_path ../data_1216/AttributionBench \
    --dataset_version ${dataset_version} \
    --template_path ../src/prompts.json \
    --model_name /ML-A800/models/flan-t5-large \
    --bs 4 \
    --split test_ood test \
    --output_dir ../inference_results/${dataset_version} \
    --max_length 2048 \
    --max_new_tokens 6 \
    --template ${template}

# fine-tuning
models=("/ML-A800/models/flan-t5-large" "/ML-A800/models/flan-t5-xl" "/ML-A800/models/flan-t5-xxl" "/ML-A800/models/flan-ul2")

for model in "${models[@]}"; do
    # 32 1e-5
    # ***************** Set parameters here *****************
    dataset_version=v3.0
    template=base_c_e
    lr=1e-5
    num_train_epoches=2
    start_gpu_index=0
    per_device_train_batch_size=1
    gas=4
    nodes=8
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
    export OUTPUT_DIR=/ML-A100/public/run/research/xiangyue/checkpoints/attribution_models/${model}-${dataset_version}-${setting}
    rm -rf $OUTPUT_DIR
    # ************************************************************************************

    export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

    # train
    torchrun --nproc_per_node $MLP_WORKER_GPU \
        --master_addr $MLP_WORKER_0_HOST \
        --node_rank $MLP_ROLE_INDEX \
        --master_port $MLP_WORKER_0_PORT \
        --nnodes $MLP_WORKER_NUM \
        ../src/train/flan-T5_train.py \
        --model_name_or_path $MODEL_DIR \
        --data_path ../data_1216/AttributionBench \
        --template ${template} \
        --template_path ../src/prompts.json \
        --dataset_version ${dataset_version} \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs $num_train_epoches \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size ${eval_bs} \
        --gradient_accumulation_steps ${gas} \
        --evaluation_strategy "no" \
        --save_strategy epoch \
        --save_total_limit 1 \
        --logging_steps 10 \
        --learning_rate $lr \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --bf16 True \
        --tf32 True \
        --report_to wandb \
        --fsdp 'full_shard auto_wrap' \
        --fsdp_transformer_layer_cls_to_wrap 'T5Block'

    # inference
    export CUDA_VISIBLE_DEVICES="0"
    python ../src/inference/run_inference.py \
        --method autoais \
        --data_path ../data_1216/AttributionBench \
        --dataset_version ${dataset_version} \
        --template_path ../src/prompts.json \
        --model_name ${OUTPUT_DIR} \
        --bs 4 \
        --split test_ood test \
        --output_dir ../inference_results/${dataset_version} \
        --max_length 2048 \
        --max_new_tokens 6 \
        --template ${template}
done