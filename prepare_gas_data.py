import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

def prepare():
    raw_data_path = 'dataset/csv_data'
    base_output_path = 'dataset/evaluation_dataset/data'
    meta_file = 'dataset/evaluation_dataset/DETECT_META.csv'
    anno_file = 'dataset/annotations.csv'
    
    target_cols = ['CHX00F003FT0101', 'CHX00F002FT0101']
    
    # 读取标注数据
    if os.path.exists(anno_file):
        anno_df = pd.read_csv(anno_file)
        # 将 file_id 转为字符串方便匹配
        anno_df['file_id'] = anno_df['file_id'].astype(str)
        print("成功加载标注数据。")
    else:
        anno_df = None
        print("未找到标注数据，将继续使用全0标签。")
    
    # 清理旧元数据，保留表头
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            header = f.readline()
        with open(meta_file, 'w') as f:
            f.write(header)
    
    meta_entries = []
    csv_files = sorted([f for f in os.listdir(raw_data_path) if f.endswith('.csv')])
    
    for col in target_cols:
        print(f"正在合并序列并同步标注: {col}...")
        all_dfs = []
        boundaries = []
        current_pos = 0
        
        for file in tqdm(csv_files):
            df = pd.read_csv(os.path.join(raw_data_path, file))
            file_id = file.replace('.csv', '')
            
            if '日期' in df.columns:
                df.rename(columns={'日期': 'date'}, inplace=True)
            elif 'date' not in df.columns:
                df.rename(columns={df.columns[0]: 'date'}, inplace=True)
            
            if col in df.columns:
                sub_df = df[['date', col]].copy()
                sub_df['label'] = 0
                
                # --- 新增：文件内局部归一化 ---
                val_mean = sub_df[col].mean()
                val_std = sub_df[col].std()
                if val_std > 1e-6:
                    sub_df[col] = (sub_df[col] - val_mean) / val_std
                else:
                    sub_df[col] = 0.0 # 处理常数序列
                # -------------------------
                
                # 应用标注
                if anno_df is not None:
                    file_anno = anno_df[anno_df['file_id'] == file_id]
                    if not file_anno.empty:
                        start_idx = int(file_anno['op_start_idx'].values[0])
                        # 处理 recovery_idx 可能为 NaN 的情况
                        end_idx_val = file_anno['recovery_idx'].values[0]
                        if pd.isna(end_idx_val):
                            end_idx = len(sub_df)
                        else:
                            end_idx = int(end_idx_val)
                        
                        # 确保索引不越界
                        start_idx = max(0, start_idx)
                        end_idx = min(len(sub_df), end_idx)
                        sub_df.iloc[start_idx:end_idx, sub_df.columns.get_loc('label')] = 1
                
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
            
            merged_df.to_csv(os.path.join(save_dir, 'merged.csv'), index=False)
            with open(os.path.join(save_dir, 'boundaries.json'), 'w') as f:
                json.dump(boundaries, f)
            
            meta_entries.append({
                'file_name': f"{col}/merged.csv",
                'dataset_name': f"Gas_{col}_Merged",
                'train_lens': int(len(merged_df) * 0.01), # 改为 1%
                'if_univariate': 'TRUE',
                'size': 'large'
            })

    if meta_entries:
        meta_df = pd.read_csv(meta_file)
        new_meta = pd.DataFrame(meta_entries)
        for c in meta_df.columns:
            if c not in new_meta.columns:
                new_meta[c] = None
        new_meta = new_meta[meta_df.columns]
        pd.concat([meta_df, new_meta], ignore_index=True).to_csv(meta_file, index=False)
        print(f"成功合并并注入标注。总异常点比例: {merged_df['label'].mean():.2%}")

if __name__ == '__main__':
    prepare()
