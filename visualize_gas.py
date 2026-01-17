import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def visualize(dataset_name, threshold):
    meta_file = 'dataset/evaluation_dataset/DETECT_META.csv'
    if not os.path.exists(meta_file):
        print(f"错误: 找不到元数据文件 {meta_file}")
        return

    meta_df = pd.read_csv(meta_file)
    
    # 查找对应的元数据行
    row = meta_df[meta_df['dataset_name'] == dataset_name]
    if row.empty:
        print(f"错误: 在 DETECT_META.csv 中找不到数据集 {dataset_name}")
        return
    
    file_rel_path = row['file_name'].values[0]
    train_lens = int(row['train_lens'].values[0])
    
    # 1. 加载原始数据
    data_path = os.path.join('dataset/evaluation_dataset/data', file_rel_path)
    if not os.path.exists(data_path):
        print(f"错误: 找不到原始数据文件 {data_path}")
        return
    
    df = pd.read_csv(data_path)
    # 统一日期列
    if '日期' in df.columns:
        df.rename(columns={'日期': 'date'}, inplace=True)
    elif 'date' not in df.columns:
        df.rename(columns={df.columns[0]: 'date'}, inplace=True)
    
    # 模型推理的是 train_lens 之后的数据
    test_df = df.iloc[train_lens:].copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    
    # 自动识别数值列
    # 排除 date 和 label 列
    value_cols = [c for c in test_df.columns if c not in ['date', 'label']]
    if not value_cols:
        print(f"错误: 在文件 {data_path} 中未找到数值列")
        return
    value_col = value_cols[0]
    values = test_df[value_col].values
    timestamps = test_df['date'].values
    
    # 2. 加载异常分数
    score_path = os.path.join('test_results', dataset_name, 'anomaly_score.npy')
    if not os.path.exists(score_path):
        print(f"错误: 找不到异常分数文件 {score_path}。请确保已运行推理脚本。")
        return
    scores = np.load(score_path)
    
    # 对齐长度（防止因窗口切分导致的极小差异）
    min_len = min(len(values), len(scores))
    values = values[:min_len]
    scores = scores[:min_len]
    timestamps = timestamps[:min_len]
    
    # 3. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # 子图1: 原始数据
    ax1.plot(timestamps, values, label=f'Original: {value_col}', color='#1f77b4', linewidth=1)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'Anomaly Detection Result: {dataset_name}', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # 子图2: 异常分数
    ax2.plot(timestamps, scores, label='Anomaly Score', color='#ff7f0e', linewidth=1)
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 4. 高亮异常区域 (阈值之上的区间设为浅红色)
    is_anomaly = scores > threshold
    if np.any(is_anomaly):
        # 寻找异常区间的起止点
        # 转换为整数数组以检测跳变
        ia_int = is_anomaly.astype(int)
        diff = np.diff(ia_int)
        
        anomaly_starts = np.where(diff == 1)[0] + 1
        if is_anomaly[0]:
            anomaly_starts = np.insert(anomaly_starts, 0, 0)
            
        anomaly_ends = np.where(diff == -1)[0]
        if is_anomaly[-1]:
            anomaly_ends = np.append(anomaly_ends, len(is_anomaly) - 1)
            
        for start, end in zip(anomaly_starts, anomaly_ends):
            ax1.axvspan(timestamps[start], timestamps[end], color='red', alpha=0.2)
            ax2.axvspan(timestamps[start], timestamps[end], color='red', alpha=0.2)
            
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片
    output_fig = os.path.join('test_results', dataset_name, 'plot.png')
    plt.savefig(output_fig, dpi=150)
    print(f"可视化图片已生成: {output_fig}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DADA Result Visualization')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Gas_CHX00F003FT0101_1012)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly threshold')
    args = parser.parse_args()
    
    visualize(args.dataset, args.threshold)
