import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
import json
import yaml
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 0. 配置并行参数 ---
# 这是你可以调整的关键参数。它决定了同时运行多少个API请求。
# 从10-20开始，根据你的API速率限制和机器性能进行调整。
MAX_WORKERS = 10 

# --- 1. 加载API配置 (与原脚本相同) ---
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

# --- 2. Prompt定义 (与原脚本相同) ---
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
    """单个API调用函数，包含错误处理。"""
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
        # 在并行环境中，打印错误比让整个程序崩溃更重要
        print(f"\nAPI 调用或解析时出错: {e}. 将返回空列表。")
        return []

# --- 3. [核心改动] 将处理单条评论的逻辑封装成一个函数 ---
def process_single_review(review_row: pd.Series) -> dict:
    """
    处理单条评论：文本分块、调用API、汇总主题。
    这是将在每个线程中独立运行的函数。
    """
    review_id = review_row['review_id']
    review_text = review_row.get('text', '')
    
    if not isinstance(review_text, str) or not review_text.strip():
        return None

    chunk_size = 2000
    text_chunks = [review_text[i:i + chunk_size] for i in range(0, len(review_text), chunk_size)]
    all_themes_for_review = set()

    for chunk in text_chunks:
        identified_themes = call_llm_api(chunk)
        all_themes_for_review.update(identified_themes)
        # 在速率限制严格的情况下，可以考虑在这里保留一个非常小的sleep
        # time.sleep(0.01) 

    result = {
        "review_id": review_id,
        "user_id": review_row['user_id'],
        "business_id": review_row['business_id'],
        "themes": list(all_themes_for_review)
    }
    return result

# --- 4. [核心改动] 主函数使用ThreadPoolExecutor进行并行处理 ---
def process_reviews_parallel():
    input_file = Path("data/processed/restaurant_reviews.json")
    output_file = Path("data/processed/review_themes.jsonl")
    
    # 断点续传逻辑保持不变
    processed_review_ids = set()
    if output_file.exists():
        print(f"发现已存在的输出文件: {output_file}。正在加载已处理的记录...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    processed_review_ids.add(json.loads(line)['review_id'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"已加载 {len(processed_review_ids)} 条已处理的评论ID。将从断点处继续。")
    
    reviews_df = pd.read_json(input_file, lines=True)
    
    # 筛选出未处理的评论
    unprocessed_reviews = reviews_df[~reviews_df['review_id'].isin(processed_review_ids)]
    
    if unprocessed_reviews.empty:
        print("所有评论均已处理完毕！")
        return

    print(f"共找到 {len(unprocessed_reviews)} 条新评论待处理。")

    # 使用追加模式打开文件，并使用线程池进行处理
    with open(output_file, 'a', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务到线程池
        # 我们将 review_row (一个Pandas Series) 作为参数传递给 process_single_review
        futures = {executor.submit(process_single_review, row) for _, row in unprocessed_reviews.iterrows()}
        
        # 使用tqdm和as_completed来处理已完成的任务，并实时显示进度条
        progress_bar = tqdm(as_completed(futures), total=len(futures), desc="并行分析评论主题")
        
        for future in progress_bar:
            try:
                result = future.result()
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    # 实时刷新文件缓冲区，确保数据被写入磁盘
                    f.flush() 
            except Exception as e:
                print(f"\n处理一个future时发生严重错误: {e}")

    print(f"\n处理完成！主题数据已保存至 {output_file}")


if __name__ == '__main__':
    process_reviews_parallel()