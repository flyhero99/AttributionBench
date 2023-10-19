



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
2.1 + retrieval.
The reason why use longlora is because the retrieval part might exceed 2048 or even the max length of 4096
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# base


source_only=ExpertQA
retriever=contriever
retrieval_k=4
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-base_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template base_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





source_only=ExpertQA
retriever=contriever
retrieval_k=6
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-base_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template base_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path






source_only=ExpertQA
retriever=contriever
retrieval_k=8
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-base_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template base_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# base



source_only=ExpertQA
retriever=dragon
retrieval_k=4
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-base_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template base_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





source_only=ExpertQA
retriever=dragon
retrieval_k=6
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-base_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template base_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path






source_only=ExpertQA
retriever=dragon
retrieval_k=8
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-base_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template base_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# w_informativeness_w_response


source_only=ExpertQA
retriever=contriever
retrieval_k=4
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=2 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-w_informativeness_w_response_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template w_informativeness_w_response_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





source_only=ExpertQA
retriever=contriever
retrieval_k=6
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=2 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-w_informativeness_w_response_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template w_informativeness_w_response_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path






source_only=ExpertQA
retriever=contriever
retrieval_k=8
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=2 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-w_informativeness_w_response_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template w_informativeness_w_response_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# w_informativeness_w_response



source_only=ExpertQA
retriever=dragon
retrieval_k=4
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=3 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-w_informativeness_w_response_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template w_informativeness_w_response_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





source_only=ExpertQA
retriever=dragon
retrieval_k=6
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=3 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-w_informativeness_w_response_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template w_informativeness_w_response_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path






source_only=ExpertQA
retriever=dragon
retrieval_k=8
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.1_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=3 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/longalpaca_13b-v2.1-template-w_informativeness_w_response_longlora-bs256-lr2e-5-gas8 \
--bs 8 --split test \
--output_dir ./inference_results_v2.1_${retriever}_top${retrieval_k}_${source_only} \
--max_length 32768 \
--max_new_tokens 6 \
--template w_informativeness_w_response_longlora \
--dataset_version v2.1 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 