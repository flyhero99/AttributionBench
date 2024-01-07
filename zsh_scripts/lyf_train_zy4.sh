# tommy server
# llama2

export HTTP_PROXY=http://100.68.173.3:3128
export HTTPS_PROXY=http://100.68.173.3:3128
export http_proxy=http://100.68.173.3:3128
export https_proxy=http://100.68.173.3:3128

export MODEL_DIR=/ML-A100/home/xiangyue/models/Llama-2-7b-hf
export WANDB_ENTITY=flyhero99
export WANDB_PROJECT=attribution-eval-v2.0

export TMPDIR=/ML-A100/home/xiangyue/lyf/tmp
export HF_HOME=/ML-A100/home/xiangyue/lyf/.cache/huggingface

# in zsh
IFS=',' read -A NUM_CUDA <<< "$CUDA_VISIBLE_DEVICES"
current_time=$(date +"%Y-%m-%d-%H:%M:%S")

# pip install jsonlines
# pip install backoff
# pip install anthropic


# stanford_new 32 1e-5
data_version=stanford_new
model=llama2_7b
# template=w_informativeness_w_response_w_longref_llama
template=base_llama_no_question
per_device_train_batch_size=1
gas=8
lr=1e-5
bs=32
setting=template-${template}-bs${bs}-lr${lr}-gas${gas}
dataset_version=stanford_new
current_time=$(date +"%Y-%m-%d-%H:%M:%S")

# make sure you want to do the deletion
# ************************************************************************************
export OUTPUT_DIR=/ML-A100/home/xiangyue/lyf/AttributionBench/checkpoints/${model}-${dataset_version}-${setting}
export CUDA_VISIBLE_DEVICES=0,1,2,3
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}
torchrun --nproc_per_node 4 --master-port 12345 ../src/train/run_train_with_question.py \
  --model_name_or_path $MODEL_DIR \
  --template ${template} \
  --template_path ../src/template.json \
  --dataset_version ${data_version} \
  --data_path /ML-A100/home/xiangyue/lyf/AttributionBench/data/hf_data \
  --num_train_samples -1 \
  --bf16 True \
  --output_dir $OUTPUT_DIR \
  --model_max_length 4096 \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps ${gas} \
  --num_train_epochs 2 \
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
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --report_to wandb \
  --gradient_checkpointing

# ************************************************************************************# ************************************************************************************# ************************************************************************************# ************************************************************************************# ************************************************************************************# ************************************************************************************# ************************************************************************************


# expertqa_rand 32 1e-5
data_version=expertqa_rand
model=llama2_7b
# template=w_informativeness_w_response_w_longref_llama
template=base_llama_no_question
per_device_train_batch_size=1
gas=8
lr=1e-5
bs=32
setting=template-${template}-bs${bs}-lr${lr}-gas${gas}
dataset_version=expertqa_rand
current_time=$(date +"%Y-%m-%d-%H:%M:%S")

# make sure you want to do the deletion
# ************************************************************************************
export OUTPUT_DIR=/ML-A100/home/xiangyue/lyf/AttributionBench/checkpoints/${model}-${dataset_version}-${setting}
export CUDA_VISIBLE_DEVICES=0,1,2,3
rm -rf $OUTPUT_DIR
# ************************************************************************************

export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}
torchrun --nproc_per_node 4 --master-port 12345 ../src/train/run_train_with_question.py \
  --model_name_or_path $MODEL_DIR \
  --template ${template} \
  --template_path ../src/template.json \
  --dataset_version ${data_version} \
  --data_path /ML-A100/home/xiangyue/lyf/AttributionBench/data/hf_data \
  --num_train_samples -1 \
  --bf16 True \
  --output_dir $OUTPUT_DIR \
  --model_max_length 4096 \
  --per_device_train_batch_size ${per_device_train_batch_size} \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps ${gas} \
  --num_train_epochs 2 \
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
  --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
  --tf32 True \
  --report_to wandb \
  --gradient_checkpointing