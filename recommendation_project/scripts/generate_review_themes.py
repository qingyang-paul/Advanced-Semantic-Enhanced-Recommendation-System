# scripts/generate_review_themes.py

import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import json

# --- 占位符：你需要自己实现的API调用函数 ---
def call_llm_api(review_text: str) -> list:
    """
    调用你的大模型API。
    
    Args:
        review_text: 评论的文本内容。
        
    Returns:
        一个包含被识别出的主题名称的列表。
        例如: ["产品与餐饮质量", "场所环境与氛围"]
    """
    # 这里是你的API调用逻辑
    # ------------------------------------------
    # 示例伪代码:
    # prompt = f"从以下评论中识别出涉及的主题，主题列表为：[员工服务与态度, ...]。只返回识别出的主题名称。\n\n评论：{review_text}"
    # response = your_api_client.chat.completions.create(model="gpt-4", messages=[...])
    # identified_themes = parse_response(response) # 你需要解析API的返回结果
    # return identified_themes
    # ------------------------------------------
    
    # 临时返回一个模拟数据用于测试
    import random
    themes = ["产品与餐饮质量", "场所环境与氛围", "效率与速度", "价格与价值"]
    return random.sample(themes, k=random.randint(1, 3))


def process_reviews_with_llm():
    """
    读取评论文件，逐条调用LLM API，并将结果保存。
    """
    input_file = Path("data/processed/restaurant_reviews.json")
    output_file = Path("data/processed/review_themes.jsonl") # 使用.jsonl扩展名
    
    print(f"正在从 {input_file} 读取评论...")
    reviews_df = pd.read_json(input_file, lines=True)
    
    # 使用tqdm来显示进度条
    with open(output_file, 'w', encoding='utf-8') as f:
        for index, row in tqdm(reviews_df.iterrows(), total=reviews_df.shape[0]):
            review_text = row['text']
            review_id = row['review_id']
            user_id = row['user_id']
            business_id = row['business_id']
            
            try:
                # 调用API
                identified_themes = call_llm_api(review_text)
                
                # 构建结果字典
                result = {
                    "review_id": review_id,
                    "user_id": user_id,
                    "business_id": business_id,
                    "themes": identified_themes
                }
                
                # 将结果以JSON Lines格式写入文件
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"处理 review_id {review_id} 时出错: {e}")
            
            # 实践中的建议：为了防止中途中断，可以每隔N条记录保存一次，
            # 并在开始时检查输出文件，实现断点续传。
            # 此外，API调用之间最好有短暂的延时以避免触发速率限制。
            # time.sleep(0.5)

    print(f"处理完成！主题数据已保存至 {output_file}")

if __name__ == '__main__':
    process_reviews_with_llm()