# python analysis_inference_results.py --data_path /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.4_for_comparision/attrbench__ML-A100_home_xiangyue_lyf_attribution-eval_tmp_llama2-7b-data_v1.4_use_testood_as_dev_with_question_bs16_lr2e-6_test_ood.json
# python analysis_inference_results.py --data_path /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.4_for_comparision/attrbench__ML-A100_home_xiangyue_lyf_attribution-eval_tmp_llama2-7b-data_v1.4_use_testood_as_dev_with_question_bs16_lr2e-6_test.json

# python analysis_inference_results.py --data_path /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.3_for_comparision/attrbench__ML-A100_home_xiangyue_lyf_attribution-eval_tmp_llama2-7b-data_v1.3_use_testood_as_dev_with_question_bs16_lr2e-6_test.json
# python analysis_inference_results.py --data_path /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.3_for_comparision/attrbench__ML-A100_home_xiangyue_lyf_attribution-eval_tmp_llama2-7b-data_v1.3_use_testood_as_dev_with_question_bs16_lr2e-6_test_ood.json





# normal finetune: no flag
# rationale fientune: w_rationle

# fewshot no rationale: --relax
# fewshot rationale: w_rationale


# for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.1_icl_2/*; do
#     if [ -f "$file" ]; then
#         if [[ $file == *"analysis.json"* ]]; then
#             continue
#         fi
#        echo "$file"
#        python analysis_inference_results.py --data_path $file --w_rationale
#     fi
# done

# for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.2_icl_2/*; do
#     if [ -f "$file" ]; then
#         if [[ $file == *"analysis.json"* ]]; then
#             continue
#         fi
#        echo "$file"
#        python analysis_inference_results.py --data_path $file --relax
#     fi
# done

# for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.2_tommy_finetuned/*; do
#     if [ -f "$file" ]; then
#         if [[ $file == *"analysis.json"* ]]; then
#             continue
#         fi
#        echo "$file"
#        python analysis_inference_results.py --data_path $file
#     fi
# done

# for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.3_tommy_finetuned/*; do
#     if [ -f "$file" ]; then
#         if [[ $file == *"analysis.json"* ]]; then
#             continue
#         fi
#        echo "$file"
#        python analysis_inference_results.py --data_path $file --w_rationale
#     fi
# done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.1_w_retrieval_k_1_dragon/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file --w_rationale
    fi
done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.1_w_retrieval_k_2_dragon/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file --w_rationale
    fi
done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.1_w_retrieval_k_3_dragon/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file --w_rationale
    fi
done




for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.1_w_retrieval_k_1_contriever/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file --w_rationale
    fi
done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.1_w_retrieval_k_2_contriever/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file --w_rationale
    fi
done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.1_w_retrieval_k_3_contriever/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file --w_rationale
    fi
done







#############





for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.2_w_retrieval_k_1_dragon/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file
    fi
done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.2_w_retrieval_k_2_dragon/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file
    fi
done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.2_w_retrieval_k_3_dragon/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file
    fi
done




for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.2_w_retrieval_k_1_contriever/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file
    fi
done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.2_w_retrieval_k_2_contriever/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file
    fi
done

for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v1.6.2_w_retrieval_k_3_contriever/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
       echo "$file"
       python analysis_inference_results.py --data_path $file
    fi
done
