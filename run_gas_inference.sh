#!/bin/bash

# 从 DETECT_META.csv 中提取包含 "Gas_" 的 dataset_name
# 排除掉表头，只取第 7 列（dataset_name）
DATASETS=$(grep "Gas_" dataset/evaluation_dataset/DETECT_META.csv | cut -d',' -f7)

if [ -z "$DATASETS" ]; then
    echo "Error: No datasets found with prefix 'Gas_' in DETECT_META.csv."
    echo "Please ensure you have run 'python prepare_gas_data.py' first."
    exit 1
fi

echo "Running inference for each sequence..."
for ds in $DATASETS
do
    echo "------------------------------------------------"
    echo "Processing $ds..."
    # 运行推理
    python -u run.py \
        --data "$ds" \
        --model ./DADA \
        --root_path ./dataset/evaluation_dataset \
        --des 'zero_shot' \
        --use_gpu True \
        --batch_size 128
done

echo "------------------------------------------------"
echo "All tasks completed! Results are in the 'test_results' folder."
