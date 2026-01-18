import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import argparse

# 配置
RAW_DATA_PATH = 'dataset/csv_data'
ANNO_FILE = 'dataset/annotations.csv'
RESULT_BASE = 'test_results'
OUTPUT_DIR = 'visualization'
GLOBAL_PLOT = os.path.join(OUTPUT_DIR, 'combined_diagnosis_global.png')
SPLIT_DIR = os.path.join(OUTPUT_DIR, 'splits')

# 传感器配置 - 直接使用 ID
TARGETS = {
    'CHX00F003FT0101': {'color': 'blue', 'name': 'CHX00F003FT0101', 'pt': 'CHX00F003PT0101'},
    'CHX00F002FT0101': {'color': 'green', 'name': 'CHX00F002FT0101', 'pt': 'CHX00F002PT0101'}
}

def main(args):
    os.makedirs(SPLIT_DIR, exist_ok=True)
    
    # 1. 加载并合并原始数据
    csv_files = sorted([f for f in os.listdir(RAW_DATA_PATH) if f.endswith('.csv')])
    all_dfs = []
    anno_df = pd.read_csv(ANNO_FILE) if os.path.exists(ANNO_FILE) else None
    
    print("正在合并原始数据并同步标注...")
    current_pos = 0
    boundaries = []
    for f in tqdm(csv_files):
        df = pd.read_csv(os.path.join(RAW_DATA_PATH, f))
        file_id = f.replace('.csv', '')
        if '日期' in df.columns: df.rename(columns={'日期': 'date'}, inplace=True)
        elif 'date' not in df.columns: df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        
        df['label'] = 0
        if anno_df is not None:
            file_anno = anno_df[anno_df['file_id'] == file_id]
            if not file_anno.empty:
                s_idx = int(file_anno['op_start_idx'].values[0])
                e_idx_val = file_anno['recovery_idx'].values[0]
                e_idx = int(e_idx_val) if not pd.isna(e_idx_val) else len(df)
                # 注入标签
                df.loc[s_idx:min(len(df), e_idx), 'label'] = 1
        
        boundaries.append({'file': f, 'start': current_pos, 'end': current_pos + len(df)})
        all_dfs.append(df)
        current_pos += len(df)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['date'] = pd.to_datetime(full_df['date'])
    total_len = len(full_df)
    train_lens = int(total_len * 0.1)
    test_df = full_df.iloc[train_lens:].reset_index(drop=True)
    
    # 2. 加载异常分数并确定阈值
    scores = {}
    thresholds = {}
    min_len = len(test_df)
    for sensor in TARGETS:
        score_path = os.path.join(RESULT_BASE, f"Gas_{sensor}_Merged", 'anomaly_score.npy')
        if os.path.exists(score_path):
            s = np.load(score_path)
            min_len = min(min_len, len(s))
            scores[sensor] = s
            
            if args.threshold is not None:
                thresholds[sensor] = args.threshold
            else:
                thresholds[sensor] = np.percentile(s, args.percentile)
            print(f"{sensor} threshold: {thresholds[sensor]:.4f} (based on {args.percentile if args.threshold is None else 'manual'} setting)")
    
    test_df = test_df.iloc[:min_len]
    for s in scores: scores[s] = scores[s][:min_len]

    # 3. 绘图函数定义
    def plot_data(df_part, score_part, file_name, save_path, is_global=False):
        fig, axes = plt.subplots(4, 1, figsize=(20, 24) if is_global else (16, 20), sharex=True)
        
        # 子图 1 & 2: 压力与流量全景
        pt_cols = [c for c in df_part.columns if 'PT' in c]
        ft_cols = [c for c in df_part.columns if 'FT' in c]
        
        for ax, cols, y_label in zip(axes[:2], [pt_cols, ft_cols], ['Pressure (MPa)', 'Flow Rate (m³/h)']):
            for col in cols:
                color, lw, zorder, alpha, label = 'gray', 0.8, 1, 1.0, None
                for s, cfg in TARGETS.items():
                    if col == cfg['pt'] or col == s:
                        color, lw, zorder, label = cfg['color'], 0.8, 5, cfg['name']
                ax.plot(df_part['date'], df_part[col], color=color, linewidth=lw, alpha=alpha, label=label, zorder=zorder)
            ax.set_ylabel(y_label)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
            ax.grid(True, linestyle='--', alpha=0.3)

        # 子图 3: 目标对比
        for s, cfg in TARGETS.items():
            axes[2].plot(df_part['date'], df_part[s], color=cfg['color'], label=cfg['name'], linewidth=1.5)
        axes[2].set_ylabel('Flow Rate (m³/h)')
        
        # 标注预测背景颜色 (去掉 GT 可视化)
        joint_anomaly = np.ones(len(df_part), dtype=bool)
        any_score_loaded = False

        for s, cfg in TARGETS.items():
            if s in score_part:
                any_score_loaded = True
                cur_scores = score_part[s]
                is_p = cur_scores > thresholds[s]
                joint_anomaly &= is_p # 累计交集
                
                if np.any(is_p):
                    diff_p = np.diff(is_p.astype(int), prepend=0, append=0)
                    for st, en in zip(np.where(diff_p==1)[0], np.where(diff_p==-1)[0]):
                        en = min(en, len(df_part)-1)
                        for ax in axes:
                            ax.axvspan(df_part['date'].iloc[st], df_part['date'].iloc[en], color=cfg['color'], alpha=0.1)

        # 额外标注：两个检测点都检测到异常的部分 (交集，红色)
        if any_score_loaded and np.any(joint_anomaly):
            diff_j = np.diff(joint_anomaly.astype(int), prepend=0, append=0)
            for st, en in zip(np.where(diff_j==1)[0], np.where(diff_j==-1)[0]):
                en = min(en, len(df_part)-1)
                for ax in axes:
                    ax.axvspan(df_part['date'].iloc[st], df_part['date'].iloc[en], color='red', alpha=0.3, label='Joint Anomaly' if (ax==axes[2] and st==np.where(diff_j==1)[0][0]) else None)

        axes[2].set_title(f"Target Sensors Comparison - {file_name}")
        axes[2].legend(loc='upper right')

        # 子图 4: 异常分数
        for s, cfg in TARGETS.items():
            if s in score_part:
                axes[3].plot(df_part['date'], score_part[s], color=cfg['color'], label=f"{cfg['name']} Score")
                axes[3].axhline(y=thresholds[s], color=cfg['color'], linestyle='--', alpha=0.5)
        axes[3].set_ylabel('Anomaly Score')
        axes[3].legend(loc='upper right', fontsize=8)
        
        for ax in axes: ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=120 if is_global else 100)
        plt.close()

    # 执行生成
    print("生成全局图...")
    plot_data(test_df, scores, "Global View", GLOBAL_PLOT, is_global=True)
    
    print("生成切分图...")
    for b in tqdm(boundaries):
        s_idx = max(0, b['start'] - train_lens)
        e_idx = min(min_len, b['end'] - train_lens)
        if e_idx <= s_idx: continue
        
        sub_df = test_df.iloc[s_idx:e_idx].reset_index(drop=True)
        sub_scores = {s: sc[s_idx:e_idx] for s, sc in scores.items()}
        plot_data(sub_df, sub_scores, b['file'], os.path.join(SPLIT_DIR, f"diag_{b['file'].replace('.csv', '.png')}"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gas Anomaly Visualization')
    parser.add_argument('--percentile', type=float, default=99.0, help='Percentile for threshold calculation (0-100)')
    parser.add_argument('--threshold', type=float, default=None, help='Manual threshold (overrides percentile)')
    args = parser.parse_args()
    main(args)

