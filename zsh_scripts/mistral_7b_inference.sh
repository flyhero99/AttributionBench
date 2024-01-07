export http_proxy=100.66.28.72:3128
export https_proxy=100.66.28.72:3128
export HTTP_PROXY=100.66.28.72:3128
export HTTPS_PROXY=100.66.28.72:3128

export TMPDIR=/ML-A800/home/xiangyue/yfli/tmp
export HF_HOME=/ML-A800/home/xiangyue/yfli/.cache/huggingface
export CUDA_VISIBLE_DEVICES=4,5,6,7


template=base_c_e
dataset_version=v3.0
export MODEL_DIR=/ML-A800/home/xiangyue/yfli/AttributionBench/checkpoints/Mistral-7B-v0.1-v3.0-template-base_c_e-bs32-lr1e-5-gas8

python ../src/inference/mistral_7b_inference.py \
--method attrbench \
--data_path ../data_1216/AttributionBench \
--dataset_version ${dataset_version} \
--template_path ../src/prompts.json \
--model_name ${MODEL_DIR} \
--bs 64 \
--split test_ood test \
--output_dir ../inference_results/${dataset_version} \
--max_length 1024 \
--max_new_tokens 6 \
--template ${template}
