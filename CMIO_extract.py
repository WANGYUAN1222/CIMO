
import os
import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# ================= 1. 配置区域 =================

DEEPSEEK_API_KEY = "sk-jMV_sR2kQcV43r9i9SquAQ"  # 替换为你的真实 Key
BASE_URL = "https://llmapi.paratera.com" 
MODEL_NAME = "Qwen3-235B-A22B-Instruct-2507"  # 支持 64K 级别长文本的 V3 模型

# 数据路径
INPUT_DIR = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/Golden_Paper" 
OUTPUT_FILE = "deepseek_table4_fulltext_reproduction.csv"

SAFE_MAX_CHARS = 120000 

# ================= 2. 强化版长文本 Prompt (修改重点) =================
# 【核心修改】：在指令和输出格式示例中，加入了 Title 和 Authors 的提取要求。
SYSTEM_PROMPT = """
You are a top-tier management literature review expert. I will provide the full text of an academic management paper (in Markdown format). 
Your task is to extract: 1. Paper metadata (Title and Authors); 2. Core "Design Propositions".

【Core Methodology: Academic Abstraction & Conciseness】
Absolutely DO NOT mechanically copy-paste long sentences from the original text! You MUST perform "theoretical abstraction":
- Strip away specific company names (e.g., instead of writing "Toyota", write "a focal firm").
- Strip away specific industry details or exact numerical values (e.g., instead of "20% cost reduction", write "decreases costs").
- Translate specific business actions into generalized management academic concepts.

【Highly Recommended Academic Vocabulary Pool】
When summarizing, please prioritize using the following high-frequency academic words:
- Intervention Verbs: choosing, maintaining, emphasizing, building, increasing, using, adopting, investing, involving.
- Outcome Verbs: increases, decreases, improves, reduces, leads to, supports, safeguards, promotes.
- Outcome Nouns: performance (operational/financial), trust, commitment, opportunism, satisfaction, innovation, agility, costs.

【Task 1: Extract Metadata】
Please extract the following from the beginning of the paper:
- Title: The full title of the paper.
- Authors: The names of the authors (separated by commas).

【Task 2: Extract Design Propositions (Strict Syntax & Structure Rules)】
An excellent top-tier paper typically demonstrates 1 to 5 core propositions. Please extract ALL core propositions and strictly output them using the following 5 fields:

1. **Theme**: Select the MOST appropriate one from the following 6 themes:
   - Theme 1: Decisions on governance mode and mechanism
   - Theme 2: Network formation and relationship initiation
   - Theme 3: Interorganizational relationships
   - Theme 4: Strategic aspects of exploiting external resources
   - Theme 5: Open innovation and interorganizational learning
   - Theme 6: Operational practices of managing external resources

2. **Context (C)**: MUST be an extremely concise prepositional phrase describing the situation.
   - ⭕ BAD: "The research investigates the situation where asset specificity is high." (Too wordy)
   - ✅ GOOD: "In the context of high asset specificity" OR "In buyer-supplier relationships"

3. **Intervention (I)**: MUST be a [Gerund Phrase starting with an -ing verb] representing the management action.
   - ⭕ BAD: "managers should choose to implement hierarchical governance" (Too wordy)
   - ✅ GOOD: "choosing hierarchical governance" OR "involving suppliers in the NPD project"

4. **Mechanism (M)**: [Logical Explanation Layer] The intermediate mechanism through which the intervention produces the outcome.
   - 💡 Note: Put the "why it works" or "through what pathway" explanations here! This acts as a buffer to ensure Intervention and Outcome remain extremely concise.
   - ✅ Example: "builds relational capital and facilitates information flow" OR "mitigates information asymmetry"

5. **Outcome (O)**: MUST be a[Third-Person Singular Verb Phrase starting with an -s/-es verb] describing the final result.
   - ⭕ BAD: "The final outcome is that it will lead to enhanced performance." (Contains redundant words)
   - ✅ GOOD: "leads to enhanced performance" OR "decreases costs and reduces delays"

【Output Format Requirements】
You MUST strictly output a valid JSON format (do not output any other explanatory markdown text):
{
    "Title": "MAKE, BUY, OR ALLY: A TRANSACTION COST THEORY META-ANALYSIS",
    "Authors": "Inge Geyskens, Jan-Benedict E. M. Steenkamp, Nirmalya Kumar",
    "propositions":[
        {
            "Theme": "Theme 1: Decisions on governance mode and mechanism",
            "Context": "In the context of high asset specificity, volume uncertainty, and behavioral uncertainty",
            "Intervention": "choosing hierarchical governance",
            "Mechanism": "increases the costs of opportunism and facilitates monitoring",
            "Outcome": "leads to enhanced performance"
        }
    ]
}
"""

# ================= 3. 核心函数 =================

def load_nested_markdown_files(root_dir):
    papers =[]
    if not os.path.exists(root_dir):
        print(f"❌ 警告: 目录不存在 - {root_dir}")
        return papers
    
    print(f"📂 正在扫描目录: {root_dir} ...")
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".md"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()[:SAFE_MAX_CHARS]
                    
                    if content:
                        papers.append({
                            "filename": filename,
                            "filepath": filepath,
                            "content": content
                        })
                except Exception as e:
                    print(f"读取文件出错 {filepath}: {e}")
                    
    print(f"✅ 共找到 {len(papers)} 个 .md 文件。")
    return papers

def parse_json_safely(text):
    try:
        return json.loads(text)
    except:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(text[start:end])
        except:
            pass
    return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=4, max=30))
def call_deepseek_api(client, input_text):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text}
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )
    return response.choices[0].message.content

def main():
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    papers = load_nested_markdown_files(INPUT_DIR)
    if not papers: return

    all_results =[]
    
    for paper in tqdm(papers, desc="🧠 全文及元数据抽取中"):
        input_text = f"论文文件名: {paper['filename']}\n论文全文内容:\n{paper['content']}"
        
        try:
            response_text = call_deepseek_api(client, input_text)
            result_dict = parse_json_safely(response_text)
            
            if result_dict:
                # 【核心修改】：先从顶层 JSON 获取论文的全局 Title 和 Authors
                paper_title = result_dict.get("Title", "Unknown Title")
                paper_authors = result_dict.get("Authors", "Unknown Authors")
                # 如果有提取出主张，则每一条主张都附加上题目和作者
                if "propositions" in result_dict and isinstance(result_dict["propositions"], list):
                    for prop in result_dict["propositions"]:
                        all_results.append({
                            "Source_File": paper["filename"],
                            "Title": paper_title,          # 新增列：题目
                            "Authors": paper_authors,      # 新增列：作者
                            "Research_Theme": prop.get("Theme", ""),
                            "Context": prop.get("Context", ""),
                            "Intervention": prop.get("Intervention", ""),
                            "Outcome": prop.get("Outcome", "")
                        })
                else:
                    print(f"\n⚠️ {paper['filename']} 提取了题目但没有找到 propositions 数组。")
            else:
                print(f"\n⚠️ {paper['filename']} 解析 JSON 失败，输出内容: {response_text[:100]}...")
                
        except Exception as e:
            print(f"\n❌ {paper['filename']} 抽取彻底失败: {e}")

        # 保护性等待，防止打爆频率限制
        time.sleep(2)

    # 4. 保存结果
    if all_results:
        # 指定列的顺序，让导出的表格更美观
        columns_order =["Source_File", "Title", "Authors", "Research_Theme", "Context", "Intervention", "Outcome"]
        df = pd.DataFrame(all_results, columns=columns_order)
        
        df.to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')
        print(f"\n🎉 抽取完成！结果已成功保存至: {OUTPUT_FILE}")
        
        print("\n【抽取结果样例】:")
        # 控制台打印时只截取部分长度，避免换行太乱
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(df.head(2))
    else:
        print("\n未能成功抽取任何数据。")

if __name__ == "__main__":
    main()