import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import difflib

# ================= 1. 配置区域 =================
# ================= 1. 配置区域 =================
DEEPSEEK_API_KEY = "sk-jMV_sR2kQcV43r9i9SquAQ"  
BASE_URL = "https://llmapi.paratera.com" 
MODEL_NAME = "Qwen3-235B-A22B-Instruct-2507" 

# 【核心修改】：使用完整的绝对路径，坚决不用 ./ 相对路径
MD_DIR = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/Golden_Paper"         
GT_CSV = "table4_ground_truth_item.csv" # 如果这个也报找不到，也换成绝对路径！
OUTPUT_JSONL = "cimo_train_dataset.jsonl"      

MAX_TRAIN_CHARS = 80000

# 微调时的文本截断（为了防止爆显存，截取前 80,000 个字符，约等于2.5万Token，足够了）
MAX_TRAIN_CHARS = 80000 

# ================= 2. 微调 System Prompt =================
# 这个 Prompt 将被硬编码到微调数据集中。模型通过学习 GT 数据，会自动学会极简抽象。
SYSTEM_PROMPT = """
You are a world-class expert in management literature systematic reviews. I will provide the full text of a management academic paper in Markdown format. 
Your task is to extract: 1. Paper metadata (Title); 2. Core "Design Propositions" based on the CIMO framework.

### CORE REQUIREMENT: MINIMALIST ACADEMIC ABSTRACTION
- DO NOT transcribe long sentences or copy segments from the text.
- You must perform high-level academic abstraction of the logical relationships.
- **Intervention**: Must be abstracted into gerund phrases (starting with -ing).
- **Outcome**: Must be abstracted into third-person singular verb phrases (e.g., "increases", "decreases", "improves", "leads to").

### OUTPUT FORMAT
You must output strictly valid JSON format. The response should contain only the JSON object, including the "Title" and a "propositions" array:
{
    "Title": "Full title of the paper",
    "propositions": [
        {
            "Theme": "Theme X: [Specific Research Theme]",
            "Context": "In the context of...",
            "Intervention": "[Gerund phrase]...",
            "Outcome": "[Third-person singular verb phrase]..."
        }
    ]
}
"""

# ================= 3. 辅助匹配函数 =================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_title_with_llm(client, text_preview):
    """让大模型从 md 文本头部提取真实标题"""
    prompt = f"""
    请从以下学术论文的开头文本中，提取出这篇论文的【确切英文标题】。
    你只需要输出标题本身，不要输出任何引号、前缀或废话！
    
    论文开头：
    {text_preview}
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def find_best_match(extracted_title, gt_titles_list):
    """使用模糊匹配在黄金标准中寻找最接近的标题"""
    if not extracted_title: return None, 0.0
    
    best_match = None
    best_ratio = 0.0
    
    ext_norm = re.sub(r'[^a-z0-9]', '', extracted_title.lower())
    
    for gt_title in gt_titles_list:
        gt_norm = re.sub(r'[^a-z0-9]', '', str(gt_title).lower())
        if not ext_norm or not gt_norm: continue
        
        # 计算相似度
        ratio = difflib.SequenceMatcher(None, ext_norm, gt_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = gt_title
            
    return best_match, best_ratio

def safe_str(val):
    return str(val).strip() if pd.notna(val) else ""

# ================= 4. 主程序 =================

def main():
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    
    # 1. 加载 GT 数据并按 Title 聚合
    print("⏳ 正在加载 Ground Truth 数据...")
    gt_df = pd.read_csv(GT_CSV)
    gt_df = gt_df.dropna(subset=['Title'])
    
    # 建立 { "论文标题": [命题1, 命题2...] } 的映射字典
    gt_dict = {}
    for title, group in gt_df.groupby('Title'):
        props =[]
        for _, row in group.iterrows():
            # 过滤掉空行
            if safe_str(row.get('Context')) or safe_str(row.get('Intervention')):
                props.append({
                    "Theme": safe_str(row.get('Research Theme')),
                    "Context": safe_str(row.get('Context')),
                    "Intervention": safe_str(row.get('Intervention')),
                    "Outcome": safe_str(row.get('Outcome'))
                })
        if props:
            gt_dict[title] = props

    gt_titles_list = list(gt_dict.keys())
    print(f"📊 黄金标准中共包含 {len(gt_titles_list)} 篇独立论文。")

    # 2. 遍历所有 md 文件并匹配
    dataset = []
    matched_count = 0
    
    md_files =[]
    for dirpath, _, filenames in os.walk(MD_DIR):
        for f in filenames:
            if f.endswith(".md"):
                md_files.append(os.path.join(dirpath, f))
                
    print(f"📂 在文件夹中共找到 {len(md_files)} 个 .md 文件。开始智能匹配...")

    for filepath in tqdm(md_files):
        with open(filepath, 'r', encoding='utf-8') as file:
            full_content = file.read().strip()
            
        if not full_content: continue
        
        # 取前 3000 个字符让 LLM 找标题
        preview = full_content[:3000]
        try:
            extracted_title = extract_title_with_llm(client, preview)
        except Exception as e:
            print(f"⚠️ 提取标题失败 ({os.path.basename(filepath)}): {e}")
            continue
            
        # 在 GT 库中找最匹配的标题
        best_gt_title, ratio = find_best_match(extracted_title, gt_titles_list)
        
        # 相似度 > 0.7 认为匹配成功 (由于去除了符号和空格，0.7 已经是非常严苛的匹配)
        if ratio > 0.70:
            matched_count += 1
            # print(f"\n[+] 匹配成功! 文件: {os.path.basename(filepath)}")
            # print(f"    提取标题: {extracted_title[:50]}...")
            # print(f"    对应GT标题: {best_gt_title[:50]}...")
            
            # ========== 构造 JSONL 一行训练数据 ==========
            # 截断全文防止 OOM
            truncated_content = full_content[:MAX_TRAIN_CHARS]
            
            # 构造标准的 Assistant 回答
            target_output = {
                "Title": best_gt_title,
                "propositions": gt_dict[best_gt_title]
            }
            
            # 组装 Swift/ShareGPT 格式
            conversation = {
                "messages":[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"论文全文内容：\n{truncated_content}"},
                    {"role": "assistant", "content": json.dumps(target_output, ensure_ascii=False, indent=2)}
                ]
            }
            dataset.append(conversation)
        else:
            pass
            # print(f"\n[-] 匹配失败. 文件: {os.path.basename(filepath)} | 提取标题: {extracted_title[:50]}...")

    # 3. 保存训练集
    print(f"\n" + "="*50)
    print(f"✅ 数据集构建完成！")
    print(f"✅ 成功将 {matched_count} 篇 .md 文件与黄金标准对齐。")
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"💾 微调文件已保存至: {OUTPUT_JSONL}")
    print("="*50)
    print("👉 下一步：你可以使用 `swift sft ... --dataset cimo_train_dataset.jsonl` 开始微调了！")

if __name__ == "__main__":
    main()