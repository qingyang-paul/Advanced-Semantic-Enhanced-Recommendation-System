# src/data_processing/create_mappings.py

import pandas as pd
import pickle
from pathlib import Path
import argparse

def generate_mappings(args):
    """
    Reads the complete filtered reviews file and creates a unique integer mapping
    for all users and businesses.
    """
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    user_map_path = output_dir / "user_id_map.pkl"
    business_map_path = output_dir / "business_id_map.pkl"
    
    print("--- 开始创建全局ID映射 ---")
    
    try:
        print(f"正在从 {input_path} 读取 'user_id' 和 'business_id' 列...")
        # We only need the ID columns, which is more memory efficient
        df_full = pd.read_json(input_path, lines=True)

    except FileNotFoundError:
        print(f"错误: 输入文件 {input_path} 未找到。")
        print("请先运行 filter_restaurants.py 生成该文件。")
        return

    columns_to_keep = ['user_id', 'business_id']
    df = df_full[columns_to_keep]

    # Create mappings from unique IDs to integer indices
    unique_users = df['user_id'].unique()
    unique_businesses = df['business_id'].unique()
    
    user_id_map = {id: i for i, id in enumerate(unique_users)}
    business_id_map = {id: i for i, id in enumerate(unique_businesses)}
    
    print(f"发现 {len(user_id_map)} 个独立用户。")
    print(f"发现 {len(business_id_map)} 个独立商户。")
    
    # Save the mappings using pickle
    print(f"正在保存用户ID映射到: {user_map_path}")
    with open(user_map_path, 'wb') as f:
        pickle.dump(user_id_map, f)
        
    print(f"正在保存商户ID映射到: {business_map_path}")
    with open(business_map_path, 'wb') as f:
        pickle.dump(business_id_map, f)
        
    print("全局ID映射创建成功！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从完整数据创建用户和商户的ID映射。")
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="../../data/processed/restaurant_reviews.json",
        help="包含所有相关评论的JSON Lines文件。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../saved_models", # Save mappings alongside the final model
        help="保存映射 (.pkl) 文件的目录。"
    )
    
    args = parser.parse_args()
    generate_mappings(args)