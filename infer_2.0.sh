export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py --method attrbench --data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf --model_name /ML-A100/home/xiangyue/lzy/attribution-eval/src/train/tmp/llama2-7b-data_v2.0_use_testood_as_dev_with_question_bs64_lr1e-5/checkpoint-259 --bs 8 --split test_ood test --output_dir ./inference_results_v2.0 --max_length 2048 --max_new_tokens 6 --template base_llama --dataset_version v2.0



export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py --method attrbench --data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf --model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-w_informativeness_w_response_llama-bs32-lr1e-5-gas8 --bs 8 --split test_ood test --output_dir ./inference_results_v2.0 --max_length 2048 --max_new_tokens 6 --template w_informativeness_w_response_llama --dataset_version v2.0


export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.0-template-base_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test_ood test \
--output_dir ./inference_results_v2.0 \
--max_length 32768 \
--max_new_tokens 6 \
--template base_longlora \
--dataset_version v2.0


export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.0-template-w_informativeness_w_response_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test_ood test \
--output_dir ./inference_results_v2.0 \
--max_length 32768 \
--max_new_tokens 6 \
--template w_informativeness_w_response_longlora \
--dataset_version v2.0





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
2.1
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 




export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-base_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test_ood test \
--output_dir ./inference_results_v2.1 \
--max_length 32768 \
--max_new_tokens 6 \
--template base_longlora \
--dataset_version v2.1


export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-w_informativeness_w_response_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test_ood test \
--output_dir ./inference_results_v2.1 \
--max_length 32768 \
--max_new_tokens 6 \
--template w_informativeness_w_response_longlora \
--dataset_version v2.1


