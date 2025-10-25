# src/data_processing/filter_restaurants.py

import pandas as pd
from pathlib import Path
import time

def filter_and_save_restaurant_reviews():
    """
    读取 business.json 和 review.json, 筛选出所有与餐厅相关的评论,
    并保存到一个新的文件中。
    """
    # --- 1. 定义文件路径 ---
    # 使用 pathlib 保证跨平台兼容性
    base_data_path = Path("../../data/unprocessed")
    output_path = Path("../../data/processed")
    
    business_file = base_data_path / "yelp_academic_dataset_business.json"
    review_file = base_data_path / "yelp_academic_dataset_review.json"
    
    # 定义输出文件
    filtered_review_file = output_path / "restaurant_reviews.json"
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("--- 开始筛选餐厅评论 ---")
    start_time = time.time()
    
    # --- 2. 筛选出所有餐厅的 business_id ---
    print(f"正在从 {business_file} 加载商户数据...")
    try:
        business_df = pd.read_json(business_file, lines=True)
    except FileNotFoundError:
        print(f"错误: 商户文件未找到 at {business_file}")
        return

    print(f"总共加载了 {len(business_df)} 个商户。")
    
    # 筛选 'categories' 列包含 "Restaurant" 的行
    # .dropna() 确保我们不会因为 'categories' 为空而出错
    # .astype(str) 将列表等内容转为字符串以便于使用 .str.contains
    restaurant_mask = business_df['categories'].astype(str).str.contains("Restaurant", case=False, na=False)
    restaurant_df = business_df[restaurant_mask]
    
    # 获取所有餐厅的唯一ID，并存入一个set中以便于快速查找
    restaurant_ids = set(restaurant_df['business_id'])
    
    if not restaurant_ids:
        print("错误: 未找到任何类别为'Restaurant'的商户。请检查数据。")
        return
        
    print(f"筛选出 {len(restaurant_ids)} 个餐厅类商户。")
    
    # --- 3. 逐块处理评论文件，进行筛选 ---
    print(f"正在从 {review_file} 逐块筛选评论...")
    
    chunk_size = 100000  # 每次处理10万条评论
    review_iterator = pd.read_json(review_file, lines=True, chunksize=chunk_size)
    
    is_first_chunk = True
    total_reviews_processed = 0
    total_reviews_kept = 0
    
    for chunk in review_iterator:
        total_reviews_processed += len(chunk)
        
        # 使用 .isin() 高效筛选 business_id 在我们餐厅ID集合中的评论
        filtered_chunk = chunk[chunk['business_id'].isin(restaurant_ids)]
        
        if not filtered_chunk.empty:
            total_reviews_kept += len(filtered_chunk)
            
            # 如果是第一个有数据的块，使用 'w' (write) 模式创建文件
            if is_first_chunk:
                filtered_chunk.to_json(filtered_review_file, orient='records', lines=True, mode='w')
                is_first_chunk = False
            # 对于后续的块，使用 'a' (append) 模式追加到文件末尾
            else:
                filtered_chunk.to_json(filtered_review_file, orient='records', lines=True, mode='a')
        
        print(f"已处理 {total_reviews_processed:,} 条评论...", end='\r')

    end_time = time.time()
    print(f"\n处理完成！总共处理了 {total_reviews_processed:,} 条评论。")
    print(f"筛选并保留了 {total_reviews_kept:,} 条餐厅相关评论。")
    print(f"已保存至: {filtered_review_file}")
    print(f"总耗时: {end_time - start_time:.2f} 秒。")

if __name__ == '__main__':
    filter_and_save_restaurant_reviews()