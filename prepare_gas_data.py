import os
import pandas as pd
from tqdm import tqdm

def prepare():
    raw_data_path = 'dataset/csv_data'
    base_output_path = 'dataset/evaluation_dataset/data'
    meta_file = 'dataset/evaluation_dataset/DETECT_META.csv'
    
    target_cols = ['CHX00F003FT0101', 'CHX00F002FT0101']
    
    # 确保输出目录存在
    for col in target_cols:
        os.makedirs(os.path.join(base_output_path, col), exist_ok=True)
    
    # 读取现有的元数据，如果没有则创建
    if os.path.exists(meta_file):
        meta_df = pd.read_csv(meta_file)
    else:
        columns = ['file_name','trend','seasonal','stationary','pattern','shifting','dataset_name','type_value','train_lens','time_steps','if_univariate','size']
        meta_df = pd.DataFrame(columns=columns)

    new_entries = []
    
    print("正在从 dataset/csv_data 提取序列并准备数据...")
    if not os.path.exists(raw_data_path):
        print(f"错误: 找不到目录 {raw_data_path}")
        return

    csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
    
    for file in tqdm(csv_files):
        file_full_path = os.path.join(raw_data_path, file)
        try:
            df = pd.read_csv(file_full_path)
        except Exception as e:
            print(f"读取文件 {file} 失败: {e}")
            continue

        # 统一日期列名
        if '日期' in df.columns:
            df.rename(columns={'日期': 'date'}, inplace=True)
        elif 'date' not in df.columns:
            # 如果没有日期列，尝试用第一列作为日期
            df.rename(columns={df.columns[0]: 'date'}, inplace=True)
        
        for col in target_cols:
            if col in df.columns:
                # 提取单变量数据
                univariate_df = df[['date', col]].copy()
                univariate_df['label'] = 0 # 默认全0标签
                
                # 保存路径：例如 dataset/evaluation_dataset/data/CHX00F003FT0101/1001.csv
                # 对应 data_provider 会寻找 root_path + /data/ + file_name
                save_rel_path = os.path.join(col, file)
                save_full_path = os.path.join(base_output_path, save_rel_path)
                
                univariate_df.to_csv(save_full_path, index=False)
                
                # 准备元数据注册信息
                # 使用 tag 区分不同的文件和传感器
                tag = f"Gas_{col}_{file.replace('.csv', '')}"
                
                # 如果元数据中已存在该 file_name，则更新或跳过
                if save_rel_path not in meta_df['file_name'].values:
                    new_entries.append({
                        'file_name': save_rel_path,
                        'dataset_name': tag,
                        'train_lens': int(len(univariate_df) * 0.2), # 默认用前20%做初始化
                        'if_univariate': 'TRUE',
                        'size': 'small'
                    })
    
    if new_entries:
        new_meta = pd.DataFrame(new_entries)
        # 确保列顺序一致
        for col in meta_df.columns:
            if col not in new_meta.columns:
                new_meta[col] = None
        new_meta = new_meta[meta_df.columns]
        
        meta_df = pd.concat([meta_df, new_meta], ignore_index=True)
        meta_df.to_csv(meta_file, index=False)
        print(f"成功注册 {len(new_entries)} 条新序列。")
    else:
        print("没有发现新序列或已全部注册。")

if __name__ == '__main__':
    prepare()
