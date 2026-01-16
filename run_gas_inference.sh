#!/bin/bash

# 1. 预处理数据并注册元数据
echo "Step 1: Preprocessing data..."
python prepare_gas_data.py

# 2. 从 DETECT_META.csv 中提取包含 "Gas_" 的 dataset_name
# 排除掉表头，只取第 7 列（dataset_name）
DATASETS=$(grep "Gas_" dataset/evaluation_dataset/DETECT_META.csv | cut -d',' -f7)

if [ -z "$DATASETS" ]; then
    echo "No datasets found with prefix 'Gas_'. Please check prepare_gas_data.py output."
    exit 1
fi

echo "Step 2: Running inference for each sequence..."
for ds in $DATASETS
do
    echo "------------------------------------------------"
    echo "Processing $ds..."
    # 运行推理，结果会保存到 test_results/Gas_...
    # 使用 zero_shot 模式，自动加载 DADA 预训练模型
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
