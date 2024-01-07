dataset_version=v3.1

for file in ../inference_results/${dataset_version}/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
        echo "$file"
        python ../analysis_inference_results.py --data_path $file
    fi
done

dataset_version=v3.2

for file in ../inference_results/${dataset_version}/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
        echo "$file"
        python ../analysis_inference_results.py --data_path $file
    fi
done

dataset_version=v3.3

for file in ../inference_results/${dataset_version}/*; do
    if [ -f "$file" ]; then
        if [[ $file == *"analysis.json"* ]]; then
            continue
        fi
        echo "$file"
        python ../analysis_inference_results.py --data_path $file
    fi
done