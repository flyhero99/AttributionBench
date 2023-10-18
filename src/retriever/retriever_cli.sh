#!/usr/bin/env bash
set -x
set -e

export HTTP_PROXY=http://100.68.173.3:3128
export HTTPS_PROXY=http://100.68.173.3:3128
export http_proxy=http://100.68.173.3:3128
export https_proxy=http://100.68.173.3:3128
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp

export CUDA_VISIBLE_DEVICES=0
python src/retriever/retrieve.py --method dragon --bs 128 --query_setting question claim --data_dir /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf --which_source ExpertQA --large_bs 500 --dataset_version v2.0


export HTTP_PROXY=http://100.68.173.3:3128
export HTTPS_PROXY=http://100.68.173.3:3128
export http_proxy=http://100.68.173.3:3128
export https_proxy=http://100.68.173.3:3128
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp

export CUDA_VISIBLE_DEVICES=1
python src/retriever/retrieve.py --method contriever --bs 128 --query_setting question claim --data_dir /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf --which_source ExpertQA --large_bs 500 --dataset_version v2.0




export HTTP_PROXY=http://100.68.173.3:3128
export HTTPS_PROXY=http://100.68.173.3:3128
export http_proxy=http://100.68.173.3:3128
export https_proxy=http://100.68.173.3:3128
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp

export CUDA_VISIBLE_DEVICES=2
python src/retriever/retrieve.py --method dragon --bs 128 --query_setting question claim --data_dir /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf --which_source Stanford-GenSearch --large_bs 500 --dataset_version v2.0


export HTTP_PROXY=http://100.68.173.3:3128
export HTTPS_PROXY=http://100.68.173.3:3128
export http_proxy=http://100.68.173.3:3128
export https_proxy=http://100.68.173.3:3128
export HF_HOME=/ML-A100/home/xiangyue/models
export TMPDIR=/ML-A100/home/xiangyue/lzy/tmp

export CUDA_VISIBLE_DEVICES=3
python src/retriever/retrieve.py --method contriever --bs 128 --query_setting question claim --data_dir /ML-A100/home/xiangyue/lyf/attribution-eval/hf_dataset/AttributionBench_branch_lyf --which_source Stanford-GenSearch --large_bs 500 --dataset_version v2.0
