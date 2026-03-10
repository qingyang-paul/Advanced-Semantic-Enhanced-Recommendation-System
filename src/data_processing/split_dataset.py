# src/data_processing/split_dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import argparse
import time

def split_data(args):
    """
    读取一个JSON Lines文件，将其分割为训练集和测试集，并保存。
    """
    # --- 1. 设置路径 ---
    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_output_path = output_dir / "train_reviews.json"
    test_output_path = output_dir / "test_reviews.json"
    
    print("--- 开始分割数据集 ---")
    start_time = time.time()

    # --- 2. 加载数据 ---
    print(f"正在从 {input_path} 加载数据...")
    try:
        df = pd.read_json(input_path, lines=True)
    except FileNotFoundError:
        print(f"错误: 输入文件未找到 at {input_path}")
        print("请先运行 filter_restaurants.py 生成该文件。")
        return
        
    print(f"成功加载 {len(df):,} 条记录。")

    # --- 3. 执行分割 ---
    print(f"正在以 {1-args.test_size:.0%}/{args.test_size:.0%} 的比例进行分割...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state  # 设置随机种子以保证结果可复现
    )
    
    print("分割完成。")
    print(f"训练集大小: {len(train_df):,} 条记录")
    print(f"测试集大小: {len(test_df):,} 条记录")

    # --- 4. 保存分割后的文件 ---
    print(f"正在保存训练集到: {train_output_path}")
    train_df.to_json(train_output_path, orient='records', lines=True)
    
    print(f"正在保存测试集到: {test_output_path}")
    test_df.to_json(test_output_path, orient='records', lines=True)

    end_time = time.time()
    print("\n所有操作成功完成！")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")

if __name__ == '__main__':
    # --- 5. 设置命令行参数解析 ---
    parser = argparse.ArgumentParser(description="将评论数据集分割为训练集和测试集。")
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="../../data/processed/restaurant_reviews.json",
        help="要分割的输入JSON Lines文件路径。"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../data/processed",
        help="保存分割后文件的目录。"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="测试集所占的比例 (例如, 0.2 代表 20%)。"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="用于保证分割可复现的随机种子。"
    )
    
    args = parser.parse_args()
    split_data(args)