for file in /ML-A100/home/xiangyue/lzy/attribution-eval/inference_results_v2.0_debug/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
        echo "$file"
        python analysis_inference_results.py --data_path $file
    fi
done