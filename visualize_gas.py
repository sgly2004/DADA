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

    # 使用 utf-8-sig 编码以处理可能存在的 BOM
    meta_df = pd.read_csv(meta_file, encoding='utf-8-sig')
    # 去除列名和值中的空格/换行符
    meta_df.columns = meta_df.columns.str.strip()
    meta_df['dataset_name'] = meta_df['dataset_name'].astype(str).str.strip()
    
    # 调试信息
    # print(f"DEBUG: Searching for '{dataset_name.strip()}'")
    # print(f"DEBUG: Available names: {meta_df['dataset_name'].tolist()[:3]}")
    
    # 查找对应的元数据行
    row = meta_df[meta_df['dataset_name'] == dataset_name.strip()]
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
        print(f"错误: 找不到异常分数文件 {score_path}")
        return
    scores = np.load(score_path)
    
    # 3. 处理拼接边界 (将拼接点前后 win_size 范围内的分数设为 0)
    boundary_file = os.path.join('dataset/evaluation_dataset/data', os.path.dirname(file_rel_path), 'boundaries.json')
    if os.path.exists(boundary_file):
        import json
        with open(boundary_file, 'r') as f:
            boundaries = json.load(f)
        
        win_size = 100 # 窗口大小
        for b in boundaries:
            # b['start'] 是拼接点
            # 注意：scores 对应的是全量数据减去 train_lens 后的部分
            # 但为了简化，我们直接在原始坐标系处理，然后截取
            pass 
        
        # 修正逻辑：在 scores 所在的索引范围内，识别边界
        # 边界点在全局索引中的位置是 b['start']
        # 转换到 scores 的索引需要减去 train_lens
        for i in range(1, len(boundaries)):
            b_idx = boundaries[i]['start'] - train_lens
            if 0 <= b_idx < len(scores):
                start = max(0, b_idx - win_size)
                end = min(len(scores), b_idx + win_size)
                scores[start:end] = 0 # 消除边界跳变干扰
    
    # 对齐长度
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
    
    # 获取真实标签
    gt_labels = test_df['label'].values[:min_len]
    
    # 子图2: 异常分数
    ax2.plot(timestamps, scores, label='Anomaly Score', color='#ff7f0e', linewidth=1)
    ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    # 4. 高亮异常区域 (阈值之上的区间设为浅红色)
    is_anomaly = scores > threshold
    if np.any(is_anomaly):
        ia_int = is_anomaly.astype(int)
        diff = np.diff(ia_int, prepend=0, append=0)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for st, en in zip(starts, ends):
            en = min(en, len(timestamps)-1)
            ax1.axvspan(timestamps[st], timestamps[en], color='red', alpha=0.2, label='Predicted' if st==starts[0] else "")
            ax2.axvspan(timestamps[st], timestamps[en], color='red', alpha=0.2)

    # 5. 高亮真实标注区间 (浅绿色)
    if np.any(gt_labels > 0):
        gt_int = gt_labels.astype(int)
        diff_gt = np.diff(gt_int, prepend=0, append=0)
        starts_gt = np.where(diff_gt == 1)[0]
        ends_gt = np.where(diff_gt == -1)[0]
        for st, en in zip(starts_gt, ends_gt):
            en = min(en, len(timestamps)-1)
            ax1.axvspan(timestamps[st], timestamps[en], color='green', alpha=0.2, label='Ground Truth' if st==starts_gt[0] else "")
            ax2.axvspan(timestamps[st], timestamps[en], color='green', alpha=0.2)
            
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片
    output_fig = os.path.join('test_results', dataset_name, 'plot_global.png')
    plt.savefig(output_fig, dpi=150)
    print(f"全局可视化图片已生成: {output_fig}")
    plt.close()

    # --- 新增：切分可视化逻辑 ---
    if os.path.exists(boundary_file):
        split_dir = os.path.join('test_results', dataset_name, 'splits')
        os.makedirs(split_dir, exist_ok=True)
        print(f"正在生成切分可视化图到: {split_dir} ...")
        
        for b in boundaries:
            file_name = b['file']
            start_idx = b['start'] - train_lens
            end_idx = b['end'] - train_lens
            
            # 确保索引在有效范围内
            if end_idx <= 0 or start_idx >= len(scores):
                continue
            
            s = max(0, start_idx)
            e = min(len(scores), end_idx)
            
            # 提取片段数据
            sub_stamps = timestamps[s:e]
            sub_values = values[s:e]
            sub_scores = scores[s:e]
            
            if len(sub_stamps) == 0:
                continue

            # 绘图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax1.plot(sub_stamps, sub_values, label='Value', color='#1f77b4')
            ax1.set_title(f"File: {file_name}", fontsize=12)
            
            ax2.plot(sub_stamps, sub_scores, label='Score', color='#ff7f0e')
            ax2.axhline(y=threshold, color='red', linestyle='--')
            
            # 高亮异常区域
            sub_is_anomaly = sub_scores > threshold
            if np.any(sub_is_anomaly):
                diff = np.diff(sub_is_anomaly.astype(int), prepend=0, append=0)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                for st, en in zip(starts, ends):
                    en = min(en, len(sub_stamps)-1)
                    ax1.axvspan(sub_stamps[st], sub_stamps[en], color='red', alpha=0.2)
                    ax2.axvspan(sub_stamps[st], sub_stamps[en], color='red', alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(split_dir, f"plot_{file_name.replace('.csv', '.png')}"))
            plt.close()
        print(f"所有切分图片已完成。")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DADA Result Visualization')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Gas_CHX00F003FT0101_1012)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Anomaly threshold')
    args = parser.parse_args()
    
    visualize(args.dataset, args.threshold)
