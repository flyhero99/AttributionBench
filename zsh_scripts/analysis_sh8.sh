dataset_versions=("v3.0" "v3.1" "v3.2" "v3.3")

method="attrbench"
# method="autoais"

for dataset_version in "${dataset_versions[@]}"; do
    for file in ../inference_results/${dataset_version}/*; do
        if [ -f "$file" ]; then
            if [[ $file == *"analysis.json"* ]]; then
                continue
            fi
            echo "$file"
            python ../analysis_inference_results.py --data_path $file --method $method
        fi
    done
done