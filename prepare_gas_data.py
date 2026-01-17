import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

def prepare():
    raw_data_path = 'dataset/csv_data'
    base_output_path = 'dataset/evaluation_dataset/data'
    meta_file = 'dataset/evaluation_dataset/DETECT_META.csv'
    
    target_cols = ['CHX00F003FT0101', 'CHX00F002FT0101']
    win_size = 100 # 对应模型 seq_len
    
    # 清理旧元数据，保留表头
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            header = f.readline()
        with open(meta_file, 'w') as f:
            f.write(header)
    
    meta_entries = []
    
    csv_files = sorted([f for f in os.listdir(raw_data_path) if f.endswith('.csv')])
    
    for col in target_cols:
        print(f"正在合并序列: {col}...")
        all_dfs = []
        boundaries = []
        current_pos = 0
        
        for file in tqdm(csv_files):
            df = pd.read_csv(os.path.join(raw_data_path, file))
            if '日期' in df.columns:
                df.rename(columns={'日期': 'date'}, inplace=True)
            elif 'date' not in df.columns:
                df.rename(columns={df.columns[0]: 'date'}, inplace=True)
            
            if col in df.columns:
                sub_df = df[['date', col]].copy()
                sub_df['label'] = 0
                
                # 记录边界：[开始索引, 结束索引]
                boundaries.append({
                    'file': file,
                    'start': current_pos,
                    'end': current_pos + len(sub_df)
                })
                
                all_dfs.append(sub_df)
                current_pos += len(sub_df)
        
        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            save_dir = os.path.join(base_output_path, col)
            os.makedirs(save_dir, exist_ok=True)
            
            save_name = f"merged.csv"
            merged_df.to_csv(os.path.join(save_dir, save_name), index=False)
            
            # 保存边界信息，供可视化使用
            with open(os.path.join(save_dir, 'boundaries.json'), 'w') as f:
                json.dump(boundaries, f)
            
            # 注册到元数据
            tag = f"Gas_{col}_Merged"
            meta_entries.append({
                'file_name': f"{col}/{save_name}",
                'dataset_name': tag,
                'train_lens': int(len(merged_df) * 0.1), # 用 10% 做初始化
                'if_univariate': 'TRUE',
                'size': 'large'
            })

    if meta_entries:
        meta_df = pd.read_csv(meta_file)
        new_meta = pd.DataFrame(meta_entries)
        # 补齐列
        for c in meta_df.columns:
            if c not in new_meta.columns:
                new_meta[c] = None
        new_meta = new_meta[meta_df.columns]
        pd.concat([meta_df, new_meta], ignore_index=True).to_csv(meta_file, index=False)
        print(f"成功合并并注册 2 条长序列。")

if __name__ == '__main__':
    prepare()
