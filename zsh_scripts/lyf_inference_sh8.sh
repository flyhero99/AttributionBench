export http_proxy=100.66.28.72:3128
export https_proxy=100.66.28.72:3128
export HTTP_PROXY=100.66.28.72:3128
export HTTPS_PROXY=100.66.28.72:3128

export TMPDIR=/ML-A800/home/xiangyue/yfli/tmp
export HF_HOME=/ML-A800/home/xiangyue/yfli/.cache/huggingface
export CUDA_VISIBLE_DEVICES=4


# 定义模板和数据集版本的数组
templates=("base_c_e" "base_c_e_r" "bace_q_c_e" "base_q_c_e_r")
dataset_versions=("v3.2" "v3.3")

# 遍历所有模板和数据集版本组合
for template in "${templates[@]}"; do
    for dataset_version in "${dataset_versions[@]}"; do
        export MODEL_DIR="/ML-A800/home/xiangyue/yfli/AttributionBench/checkpoints/llama2-7b-hf-${dataset_version}-template-${template}-bs32-lr1e-5-gas8"

        echo "Running inference for template: $template and dataset_version: $dataset_version"

        python ../src/inference/run_inference.py \
        --method attrbench \
        --data_path ../data_1216/AttributionBench \
        --dataset_version ${dataset_version} \
        --template_path ../src/prompts.json \
        --model_name ${MODEL_DIR} \
        --bs 4 \
        --split test_ood test \
        --output_dir ../inference_results/${dataset_version} \
        --max_length 2048 \
        --max_new_tokens 6 \
        --template ${template}
    done
done
