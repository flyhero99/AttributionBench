export TMPDIR=/ML-A100/home/xiangyue/lyf/tmp
export HF_HOME=/ML-A100/home/xiangyue/lyf/.cache/huggingface

CUDA_VISIBLE_DEVICES=0,1 python ../src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/AttributionBench/data/hf_data \
--dataset_version v2.1 \
--template_path ../src/template.json \
--model_name /ML-A100/home/xiangyue/lyf/AttributionBench/checkpoints/llama2_7b-v2.1-template-base_llama-bs32-lr1e-5-gas8 \
--bs 8 \
--split test_ood test \
--output_dir /ML-A100/home/xiangyue/lyf/AttributionBench/inference_results/v2.1_all \
--max_length 4096 \
--max_new_tokens 6 \
--template base_llama
