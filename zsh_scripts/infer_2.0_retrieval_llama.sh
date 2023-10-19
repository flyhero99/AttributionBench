



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
2.0 + retrieval.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# base


source_only=ExpertQA
retriever=contriever
retrieval_k=1
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=3 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-base_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template base_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





source_only=ExpertQA
retriever=contriever
retrieval_k=2
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=3 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-base_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template base_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path






source_only=ExpertQA
retriever=contriever
retrieval_k=3
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=3 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-base_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template base_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# base



source_only=ExpertQA
retriever=dragon
retrieval_k=1
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-base_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template base_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





source_only=ExpertQA
retriever=dragon
retrieval_k=2
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-base_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template base_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path






source_only=ExpertQA
retriever=dragon
retrieval_k=3
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=1 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-base_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template base_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# w_informativeness_w_response


source_only=ExpertQA
retriever=contriever
retrieval_k=1
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=2 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-w_informativeness_w_response_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template w_informativeness_w_response_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





source_only=ExpertQA
retriever=contriever
retrieval_k=2
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=2 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-w_informativeness_w_response_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template w_informativeness_w_response_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path






source_only=ExpertQA
retriever=contriever
retrieval_k=3
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=2 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-w_informativeness_w_response_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template w_informativeness_w_response_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path








# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# w_informativeness_w_response



source_only=ExpertQA
retriever=dragon
retrieval_k=1
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-w_informativeness_w_response_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template w_informativeness_w_response_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





source_only=ExpertQA
retriever=dragon
retrieval_k=2
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-w_informativeness_w_response_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template w_informativeness_w_response_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path






source_only=ExpertQA
retriever=dragon
retrieval_k=3
retrieval_path="/ML-A100/home/xiangyue/lzy/attribution-eval/${retriever}_v2.0_question|claim_${source_only}/retrieval_clean_${retriever}.jsonl"


# check data version
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp
CUDA_VISIBLE_DEVICES=0 python src/inference/run_inference.py \
--method attrbench \
--data_path /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf \
--model_name /ML-A100/home/xiangyue/lzy/attribution-eval/checkpoint/llama2_7b-v2.0-template-w_informativeness_w_response_llama-bs32-lr1e-5-gas8-initialization-False \
--bs 8 --split test \
--output_dir ./inference_results_v2.0_${retriever}_top${retrieval_k}_${source_only} \
--max_length 4096 \
--max_new_tokens 6 \
--template w_informativeness_w_response_llama \
--dataset_version v2.0 \
--source_only $source_only \
--retrieval_k ${retrieval_k} \
--retrieval_path $retrieval_path





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
