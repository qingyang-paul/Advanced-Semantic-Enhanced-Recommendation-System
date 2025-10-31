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
    # 确保输出目录存在，这是一个好习惯
    output_dir.mkdir(parents=True, exist_ok=True) 
    
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
    # --- 修改之处 ---
    # 在聚合前，先用 drop() 排除掉非数值的 user_id 列
    business_profiles = full_df.drop(columns=['user_id']).groupby('business_id').mean()
    # --- 修改结束 ---
    business_output_path = output_dir / "business_theme_profiles.jsonl"
    # 1. 将 business_id 从索引变回普通列
    business_profiles.reset_index(inplace=True)
    # 2. 保存为 JSON Lines 格式 (每行一个JSON对象)
    business_profiles.to_json(
        business_output_path, 
        orient='records',    # 将每行转换为一个字典
        lines=True,          # 每个字典写在一行
        force_ascii=False    # 保证汉字可读
    )
    print(f"商家画像已保存至 {business_output_path}")

    # 3. 按 user_id 聚合，计算每个主题的平均提及率
    print("正在聚合用户主题画像...")
    # --- 修改之处 ---
    # 同理，在聚合前排除掉 business_id 列
    user_profiles = full_df.drop(columns=['business_id']).groupby('user_id').mean()
    # --- 修改结束 ---
    user_output_path = output_dir / "user_theme_profiles.jsonl"
    # 1. 将 user_id 从索引变回普通列
    user_profiles.reset_index(inplace=True)
    # 2. 保存为 JSON Lines 格式
    user_profiles.to_json(
        user_output_path, 
        orient='records', 
        lines=True, 
        force_ascii=False
    )
    print(f"用户画像已保存至 {user_output_path}")
    
if __name__ == '__main__':
    aggregate_theme_features()