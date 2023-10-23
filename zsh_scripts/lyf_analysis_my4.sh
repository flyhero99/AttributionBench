for file in /ML-A100/home/xiangyue/lyf/AttributionBench/inference_results/v2.3/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
        echo "$file"
        python ../analysis_inference_results.py --data_path $file
    fi
done