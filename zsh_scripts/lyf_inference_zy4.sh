export TMPDIR=/ML-A100/home/xiangyue/lyf/tmp
export HF_HOME=/ML-A100/home/xiangyue/lyf/.cache/huggingface

CUDA_VISIBLE_DEVICES=0,1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/AttributionBench/data/hf_data \
--dataset_version v2.0 \
--model_name  \
--bs 8 \
--split test_ood test \
--output_dir  \
--max_length 4096 \
--max_new_tokens 6
# --template w_informativeness_w_response_llama