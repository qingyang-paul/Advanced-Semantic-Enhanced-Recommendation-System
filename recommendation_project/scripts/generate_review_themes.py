# scripts/generate_review_themes.py

import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import json
import yaml  # Using PyYAML for config files
import openai

# --- 1. Load API Config from File (Your Request #1) ---
try:
    with open("configs/api_config.yaml", 'r') as f:
        api_config = yaml.safe_load(f)['openai']
    API_KEY = api_config.get("api_key")
    BASE_URL = api_config.get("base_url")
    MODEL_ID = api_config.get("model_id")
except (FileNotFoundError, KeyError) as e:
    raise ValueError(f"错误: 请创建并正确填写 configs/api_config.yaml 文件。 详情: {e}")

if not all([API_KEY, BASE_URL, MODEL_ID]) or "YOUR_API_KEY_HERE" in API_KEY:
    raise ValueError("请在 configs/api_config.yaml 中提供有效的 API 密钥、URL 和模型ID。")

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# --- (Prompt definition is the same) ---
THEMES_LIST = [
    "员工服务与态度", "产品与餐饮质量", "场所环境与氛围", "效率与速度", 
    "问题解决与管理", "地理位置与便利性", "价格与价值", "餐饮/专业服务质量", 
    "忠诚度与长期关系", "额外体验与设施"
]
SYSTEM_PROMPT = f"""
你是一名专业的餐厅评论分析师。你的任务是仔细阅读一份用户评论，并从以下预定义的主题列表中，识别出评论中明确提及或强烈暗示的所有主题。
# 预定义主题列表:
{', '.join(THEMES_LIST)}
# 输出要求:
请严格按照以下JSON格式返回你的分析结果，不要添加任何额外的解释或文字。
{{
  "themes": ["识别出的主题1", "识别出的主题2", ...]
}}
"""

def call_llm_api(review_text_chunk: str) -> list:
    # (The API call function itself is mostly the same, but with improved error handling)
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": review_text_chunk}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256
        )
        response_content = response.choices[0].message.content
        result_json = json.loads(response_content)
        identified_themes = result_json.get("themes", [])
        return identified_themes if isinstance(identified_themes, list) else []
    except Exception as e:
        print(f"\nAPI 调用或解析时出错: {e}. 将返回空列表。")
        return []


def process_reviews_with_llm():
    input_file = Path("data/processed/restaurant_reviews.json")
    output_file = Path("data/processed/review_themes.jsonl")
    
    # --- 3. 实现断点续传 (Your Request #3) ---
    processed_review_ids = set()
    if output_file.exists():
        print(f"发现已存在的输出文件: {output_file}。正在加载已处理的记录...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_review_ids.add(json.loads(line)['review_id'])
                except json.JSONDecodeError:
                    continue # Skip corrupted lines
        print(f"已加载 {len(processed_review_ids)} 条已处理的评论ID。将从断点处继续。")
    
    reviews_df = pd.read_json(input_file, lines=True)
    
    # 打开文件以追加模式 ('a')，这样就不会覆盖已有内容
    with open(output_file, 'a', encoding='utf-8') as f:
        # 使用tqdm包装DataFrame的迭代器
        progress_bar = tqdm(reviews_df.iterrows(), total=reviews_df.shape[0], desc="分析评论主题")
        for index, row in progress_bar:
            review_id = row['review_id']
            
            # 如果此ID已经处理过，则跳过
            if review_id in processed_review_ids:
                continue

            review_text = row.get('text', '')
            if not isinstance(review_text, str) or not review_text.strip():
                continue

            # --- 2. 截断长文本并分块处理 (Your Request #2) ---
            chunk_size = 2000  # 每个API请求处理的字符数
            text_chunks = [review_text[i:i + chunk_size] for i in range(0, len(review_text), chunk_size)]
            all_themes_for_review = set() # 使用set自动去重

            for chunk in text_chunks:
                identified_themes = call_llm_api(chunk)
                all_themes_for_review.update(identified_themes)
                # 在每个块之间稍微暂停，可以进一步降低API速率限制风险
                time.sleep(0.01)

            result = {
                "review_id": review_id,
                "user_id": row['user_id'],
                "business_id": row['business_id'],
                "themes": list(all_themes_for_review) # 将set转回list进行保存
            }
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            # 更新进度条的描述
            progress_bar.set_postfix({"Last ID": review_id[-5:]})

    print(f"\n处理完成！主题数据已保存至 {output_file}")

if __name__ == '__main__':
    process_reviews_with_llm()