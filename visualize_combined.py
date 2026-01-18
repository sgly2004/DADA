import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

# 配置
RAW_DATA_PATH = 'dataset/csv_data'
RESULT_BASE = 'test_results'
OUTPUT_FILE = 'test_results/gas_combined_diagnosis.png'

# 传感器配置
TARGETS = {
    'CHX00F003FT0101': {'color': 'blue', 'name': 'Wushen Flow', 'pt': 'CHX00F003PT0101'},
    'CHX00F002FT0101': {'color': 'green', 'name': 'Etoke Flow', 'pt': 'CHX00F002PT0101'}
}

def main():
    # 1. 加载并合并原始数据
    csv_files = sorted([f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')])
    all_dfs = []
    print("正在合并原始数据用于全景展示...")
    for f in tqdm(csv_files) if 'tqdm' in globals() else csv_files:
        df = pd.read_csv(os.path.join(RAW_DATA_PATH, f))
        if '日期' in df.columns: df.rename(columns={'日期': 'date'}, inplace=True)
        elif 'date' not in df.columns: df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        all_dfs.append(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['date'] = pd.to_datetime(full_df['date'])
    total_len = len(full_df)
    train_lens = int(total_len * 0.1) # 对应 prepare_gas_data.py 中的比例
    
    # 截取测试部分
    test_df = full_df.iloc[train_lens:].reset_index(drop=True)
    
    # 2. 加载异常分数
    scores = {}
    thresholds = {}
    min_len = len(test_df)
    
    for sensor in TARGETS:
        score_path = os.path.join(RESULT_BASE, f"Gas_{sensor}_Merged", 'anomaly_score.npy')
        if os.path.exists(score_path):
            s = np.load(score_path)
            min_len = min(min_len, len(s))
            scores[sensor] = s
            thresholds[sensor] = np.percentile(s, 99)
            print(f"{sensor} 99th percentile: {thresholds[sensor]:.4f}")
        else:
            print(f"警告: 找不到 {sensor} 的分数文件")

    # 对齐所有数据长度
    test_df = test_df.iloc[:min_len]
    for sensor in scores:
        scores[sensor] = scores[sensor][:min_len]

    # 3. 绘图 (4个子图)
    fig, axes = plt.subplots(4, 1, figsize=(20, 24), sharex=True)
    
    # 子图 1: 压力全景 (PT 结尾的列)
    pt_cols = [c for c in full_df.columns if 'PT' in c]
    for col in pt_cols:
        color = 'gray'
        alpha = 0.2
        linewidth = 0.5
        label = None
        # 如果是目标对应的 PT，稍微突出一点
        for sensor, cfg in TARGETS.items():
            if col == cfg['pt']:
                color = cfg['color']
                alpha = 0.6
                linewidth = 1.0
                label = f"{cfg['name']} Press"
        axes[0].plot(test_df['date'], test_df[col], color=color, alpha=alpha, linewidth=linewidth, label=label)
    axes[0].set_title('Pressure Context (All PT Sensors)', fontsize=16)
    axes[0].set_ylabel('Pressure (MPa)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # 子图 2: 流量全景 (FT 结尾的列)
    ft_cols = [c for c in full_df.columns if 'FT' in c]
    for col in ft_cols:
        color = 'gray'
        alpha = 0.2
        linewidth = 0.5
        label = None
        # 如果是目标 FT，稍微突出一点
        if col in TARGETS:
            color = TARGETS[col]['color']
            alpha = 0.6
            linewidth = 1.0
            label = TARGETS[col]['name']
        axes[1].plot(test_df['date'], test_df[col], color=color, alpha=alpha, linewidth=linewidth, label=label)
    axes[1].set_title('Flow Context (All FT Sensors)', fontsize=16)
    axes[1].set_ylabel('Flow Rate (m³/h)', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 子图 3: 两个关键流量
    for sensor, cfg in TARGETS.items():
        axes[2].plot(test_df['date'], test_df[sensor], color=cfg['color'], label=cfg['name'], linewidth=2.0)
    axes[2].set_title('Target Flow Sensors Comparison', fontsize=16)
    axes[2].set_ylabel('Flow Rate (m³/h)', fontsize=12)
    axes[2].legend(loc='upper right')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    # 子图 4: 异常分数
    for sensor, cfg in TARGETS.items():
        if sensor in scores:
            axes[3].plot(test_df['date'], scores[sensor], color=cfg['color'], label=f"{cfg['name']} Score", alpha=0.8)
            axes[3].axhline(y=thresholds[sensor], color=cfg['color'], linestyle='--', alpha=0.6, label=f"{cfg['name']} 99th Thr")
    axes[3].set_title('Anomaly Scores & 99th Percentile Thresholds', fontsize=16)
    axes[3].set_ylabel('Score', fontsize=12)
    axes[3].set_xlabel('Date', fontsize=12)
    axes[3].legend(loc='upper right')
    axes[3].grid(True, linestyle='--', alpha=0.5)

    # 处理边界抑制线 (从 boundaries.json 读取)
    boundary_file = os.path.join('dataset/evaluation_dataset/data', 'CHX00F003FT0101', 'boundaries.json')
    if os.path.exists(boundary_file):
        with open(boundary_file, 'r') as f:
            boundaries = json.load(f)
        for i in range(1, len(boundaries)):
            b_idx = boundaries[i]['start'] - train_lens
            if 0 <= b_idx < min_len:
                for ax in axes:
                    ax.axvline(x=test_df['date'].iloc[b_idx], color='black', linestyle=':', alpha=0.2)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=120)
    print(f"组合诊断图已生成: {OUTPUT_FILE}")

if __name__ == '__main__':
    from tqdm import tqdm
    main()
