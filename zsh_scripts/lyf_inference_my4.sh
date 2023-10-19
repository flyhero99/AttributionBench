export TMPDIR=/ML-A100/home/xiangyue/lyf/tmp
export HF_HOME=/ML-A100/home/xiangyue/lyf/.cache/huggingface

CUDA_VISIBLE_DEVICES=0,1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--dataset_version v2.0 \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-w_informativeness_w_response_llama-bs32-lr1e-5-gas8 \
--bs 8 \
--split test_ood test \
--output_dir /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v2.0_debug \
--max_length 4096 \
--max_new_tokens 6
# --template w_informativeness_w_response_llama