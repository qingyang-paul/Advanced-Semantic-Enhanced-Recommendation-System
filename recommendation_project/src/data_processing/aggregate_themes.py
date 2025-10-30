# src/data_processing/aggregate_themes.py

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

def aggregate_theme_features():
    """
    读取LLM生成的主题数据，并为每个用户和商家创建平均主题向量（画像）。
    """
    input_file = Path("data/processed/review_themes.jsonl")
    output_dir = Path("data/processed")
    
    print(f"正在从 {input_file} 读取主题数据...")
    themes_df = pd.read_json(input_file, lines=True)

    # 1. 将主题列表转换为多热编码（Multi-hot Encoding）的向量
    mlb = MultiLabelBinarizer()
    theme_vectors = mlb.fit_transform(themes_df['themes'])
    theme_df = pd.DataFrame(theme_vectors, columns=mlb.classes_, index=themes_df.index)
    
    # 将编码后的向量和ID信息合并
    full_df = pd.concat([themes_df[['user_id', 'business_id']], theme_df], axis=1)
    
    print("所有主题:", list(mlb.classes_))
    
    # 2. 按 business_id 聚合，计算每个主题的平均提及率
    print("正在聚合商家主题画像...")
    business_profiles = full_df.groupby('business_id').mean()
    business_output_path = output_dir / "business_theme_profiles.json"
    business_profiles.to_json(business_output_path, orient='index')
    print(f"商家画像已保存至 {business_output_path}")

    # 3. 按 user_id 聚合，计算每个主题的平均提及率
    print("正在聚合用户主题画像...")
    user_profiles = full_df.groupby('user_id').mean()
    user_output_path = output_dir / "user_theme_profiles.json"
    user_profiles.to_json(user_output_path, orient='index')
    print(f"用户画像已保存至 {user_output_path}")
    
if __name__ == '__main__':
    aggregate_theme_features()