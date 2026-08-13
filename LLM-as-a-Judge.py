import os
import json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import difflib

# ================= 1. 配置区域 =================
DEEPSEEK_API_KEY = "sk-jMV_sR2kQcV43r9i9SquAQ"  # 你的 API Key
BASE_URL = "https://llmapi.paratera.com" 
MODEL_NAME = "Qwen3-235B-A22B-Instruct-2507" 

LLM_FILE = 'deepseek_table4_fulltext_reproduction.csv'        
GT_FILE = 'table4_ground_truth_item.csv'              
OUTPUT_EVAL_FILE = 'cimo_full_evaluation_results.csv' 

# ================= 2. 全要素裁判 Prompt =================
JUDGE_PROMPT = """
你是一位严谨的管理学顶级期刊评审专家。
你的任务是评估【AI提取的结论】是否成功还原了【原作者的黄金标准】。
你必须对 CIMO 逻辑中的 Context (情境), Intervention (干预), Outcome (结果) 进行【独立拆解评估】。

【评估准则】
1. Context (C) 匹配：AI提取的情境是否与黄金标准属于同一特定环境？（允许AI更详细或使用同义词）
2. Intervention (I) 匹配：核心管理动作的实质含义是否相同？
3. Outcome (O) 匹配：结果的方向（增/减、好/坏）和核心变量是否一致？绝对不能将反义词判为匹配！
4. 综合匹配 (overall_match)：只有当核心的因果逻辑链 (C -> I -> O) 整体上没有学术冲突时，综合匹配才为 true。

【比对数据】
--- 黄金标准 (Ground Truth) ---[Context]: {gt_c}
[Intervention]: {gt_i}
[Outcome]: {gt_o}

--- AI 提取结果 (Prediction) ---
[Context]: {llm_c}
[Intervention]: {llm_i}
[Outcome]: {llm_o}

【输出格式】
你必须且只能输出一个合法的 JSON 对象！键名必须全小写！
{
    "c_match": true,
    "c_reason": "情境匹配理由",
    "i_match": true,
    "i_reason": "干预匹配理由",
    "o_match": false,
    "o_reason": "结果匹配理由",
    "overall_match": false,
    "overall_reason": "综合判定理由"
}
"""

def is_title_match(t1, t2):
    if pd.isna(t1) or pd.isna(t2): return False
    t1_norm = re.sub(r'[^a-z0-9\s]', '', str(t1).lower()).strip()
    t2_norm = re.sub(r'[^a-z0-9\s]', '', str(t2).lower()).strip()
    if not t1_norm or not t2_norm: return False
    if t1_norm in t2_norm or t2_norm in t1_norm: return True
    return difflib.SequenceMatcher(None, t1_norm, t2_norm).ratio() > 0.85

def safe_str(val):
    return str(val) if pd.notna(val) else ""

def normalize_json(raw_dict):
    """将所有键转为小写，将字符串的 true/false 强制转为布尔值"""
    if not isinstance(raw_dict, dict): return {}
    clean_dict = {}
    for k, v in raw_dict.items():
        key = str(k).lower().strip() 
        if isinstance(v, str):
            if v.lower() == 'true': v = True
            elif v.lower() == 'false': v = False
        clean_dict[key] = v
    return clean_dict

def parse_json_safely(text):
    try:
        data = json.loads(text)
        return normalize_json(data)
    except:
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != -1:
                data = json.loads(text[start:end])
                return normalize_json(data)
        except Exception as e:
            print(f"\n[Debug] JSON 解析失败! 大模型输出原文: {text[:150]}")
    return None

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def call_cimo_judge(client, gt_row, llm_row):
    # 【核心修复】：放弃 .format()，改用 .replace() 避免 JSON 大括号引发的 KeyError
    user_content = JUDGE_PROMPT.replace("{gt_c}", safe_str(gt_row.get('Context', ''))) \
                               .replace("{gt_i}", safe_str(gt_row.get('Intervention', ''))) \
                               .replace("{gt_o}", safe_str(gt_row.get('Outcome', ''))) \
                               .replace("{llm_c}", safe_str(llm_row.get('Context', ''))) \
                               .replace("{llm_i}", safe_str(llm_row.get('Intervention', ''))) \
                               .replace("{llm_o}", safe_str(llm_row.get('Outcome', '')))
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": user_content}],
        temperature=0.0
    )
    
    response_text = response.choices[0].message.content
    res_json = parse_json_safely(response_text)
    
    if res_json is not None:
        return res_json
    else:
        raise ValueError("JSON提取彻底失败")

# ================= 3. 主程序 =================
def main():
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    
    try:
        llm_df = pd.read_csv(LLM_FILE)
        gt_df = pd.read_csv(GT_FILE)
        llm_df.columns = llm_df.columns.str.strip()
        gt_df.columns = gt_df.columns.str.strip()
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    gt_df = gt_df.dropna(subset=['Context', 'Intervention', 'Outcome', 'Title'], how='all')

    print(f"\n🚀 开始全要素 CIMO 裁判评估 (共 {len(gt_df)} 条 GT)...")
    
    evaluation_logs =[]
    stats = {'total_gt': len(gt_df), 'c_matched': 0, 'i_matched': 0, 'o_matched': 0, 'overall_matched': 0}

    for index, gt_row in tqdm(gt_df.iterrows(), total=len(gt_df)):
        gt_title = gt_row.get('Title', '')
        if pd.isna(gt_title): continue

        llm_candidates = llm_df[llm_df['Title'].apply(lambda x: is_title_match(gt_title, x))]
        
        log_entry = {
            "Paper_Title": gt_title,
            "GT_Context": safe_str(gt_row.get('Context')),
            "GT_Intervention": safe_str(gt_row.get('Intervention')),
            "GT_Outcome": safe_str(gt_row.get('Outcome'))
        }

        if llm_candidates.empty:
            log_entry.update({
                "LLM_Context": "N/A", "LLM_Intervention": "N/A", "LLM_Outcome": "N/A",
                "C_Match": False, "I_Match": False, "O_Match": False, "Overall_Match": False,
                "Reason": "模型未提取该论文"
            })
            evaluation_logs.append(log_entry)
            continue

        best_match_result = None
        best_llm_row = None
        error_messages =[]
        
        for _, llm_row in llm_candidates.iterrows():
            try:
                judge_res = call_cimo_judge(client, gt_row, llm_row)
                if best_match_result is None:
                    best_match_result = judge_res
                    best_llm_row = llm_row
                
                if judge_res.get('overall_match') is True:
                    best_match_result = judge_res
                    best_llm_row = llm_row
                    break
            except Exception as e:
                error_messages.append(str(e))
                
        # 写入判定结果
        if best_match_result:
            c_val = best_match_result.get('c_match') is True
            i_val = best_match_result.get('i_match') is True
            o_val = best_match_result.get('o_match') is True
            overall_val = best_match_result.get('overall_match') is True

            log_entry.update({
                "LLM_Context": safe_str(best_llm_row.get('Context')),
                "LLM_Intervention": safe_str(best_llm_row.get('Intervention')),
                "LLM_Outcome": safe_str(best_llm_row.get('Outcome')),
                "C_Match": c_val, "C_Reason": best_match_result.get('c_reason', ''),
                "I_Match": i_val, "I_Reason": best_match_result.get('i_reason', ''),
                "O_Match": o_val, "O_Reason": best_match_result.get('o_reason', ''),
                "Overall_Match": overall_val, "Overall_Reason": best_match_result.get('overall_reason', '')
            })
            
            if c_val: stats['c_matched'] += 1
            if i_val: stats['i_matched'] += 1
            if o_val: stats['o_matched'] += 1
            if overall_val: stats['overall_matched'] += 1
        else:
            log_entry.update({
                "Overall_Match": False, 
                "Overall_Reason": f"API调用失败记录: {'; '.join(error_messages)[:100]}"
            })
            
        evaluation_logs.append(log_entry)

    # ================= 4. 输出细粒度结果 =================
    eval_df = pd.DataFrame(evaluation_logs)
    eval_df.to_csv(OUTPUT_EVAL_FILE, index=False, encoding='utf_8_sig')
    
    t = stats['total_gt']
    
    print("\n" + "="*55)
    print(f"📊 【全要素 CIMO 逻辑评估报告】 📊")
    print("="*55)
    print(f"黄金标准总条数 (GT): {t}")
    if t > 0:
        print(f"🏆 综合逻辑召回率 (Overall Match): {stats['overall_matched']/t:.2%} ({stats['overall_matched']}/{t})")
        print("-" * 55)
        print(f"🧩 【细粒度要素召回率 (Component Recall)】")
        print(f"   📍 Context (情境) 准确率:      {stats['c_matched']/t:.2%} ({stats['c_matched']}/{t})")
        print(f"   📍 Intervention (干预) 准确率: {stats['i_matched']/t:.2%} ({stats['i_matched']}/{t})")
        print(f"   📍 Outcome (结果) 准确率:      {stats['o_matched']/t:.2%} ({stats['o_matched']}/{t})")
    print("="*55)
    print(f"💡 详细的 C、I、O 对比日志与裁判理由已保存至: {OUTPUT_EVAL_FILE}")

if __name__ == "__main__":
    main()