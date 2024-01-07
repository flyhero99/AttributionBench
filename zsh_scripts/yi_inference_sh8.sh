export http_proxy=100.66.28.72:3128
export https_proxy=100.66.28.72:3128
export HTTP_PROXY=100.66.28.72:3128
export HTTPS_PROXY=100.66.28.72:3128

export TMPDIR=/ML-A800/home/xiangyue/yfli/tmp
export HF_HOME=/ML-A800/home/xiangyue/yfli/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1,2,3


template=base_c_e
dataset_version=v3.0
export MODEL_DIR=/ML-A800/home/xiangyue/yfli/AttributionBench/checkpoints/Yi-6b-Chat-v3.0-template-base_c_e-bs32-lr1e-5-gas8/checkpoint-832

python ../src/inference/run_inference.py \
--method attrbench \
--data_path ../data_1216/AttributionBench \
--dataset_version ${dataset_version} \
--template_path ../src/prompts.json \
--model_name ${MODEL_DIR} \
--bs 16 \
--split test_ood test \
--output_dir ../inference_results/${dataset_version} \
--max_length 4096 \
--max_new_tokens 6 \
--template ${template}