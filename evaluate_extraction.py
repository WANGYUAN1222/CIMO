import os
import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
from modelscope import snapshot_download
import difflib

# 1. 使用 ModelScope 从国内极速下载模型到本地缓存
print("正在从 ModelScope 下载模型...")
# 如果你已经有了本地路径，下面这行可以注释掉
model_dir = snapshot_download('BAAI/bge-large-en-v1.5', cache_dir='./')

# ================= 1. 配置区域 =================

LLM_FILE = 'deepseek_table4_fulltext_reproduction.csv'   # 大模型提取的结果 
GT_FILE = 'table4_ground_truth_item.csv'                 # 黄金标准表 
SIMILARITY_THRESHOLD = 0.9                               # 语义相似度阈值

# ================= 2. 辅助函数 =================

def is_title_match(t1, t2):
    """标题鲁棒匹配函数"""
    if pd.isna(t1) or pd.isna(t2):
        return False
    t1_norm = re.sub(r'[^a-z0-9\s]', '', str(t1).lower()).strip()
    t2_norm = re.sub(r'[^a-z0-9\s]', '', str(t2).lower()).strip()
    if not t1_norm or not t2_norm:
        return False
    if t1_norm in t2_norm or t2_norm in t1_norm:
        return True
    ratio = difflib.SequenceMatcher(None, t1_norm, t2_norm).ratio()
    return ratio > 0.85

def safe_str(val):
    """安全转换为字符串，处理 NaN"""
    return str(val) if pd.notna(val) else ""

def concat_cio(row):
    """将 C, I, O 拼接成一段完整的语义文本"""
    c = safe_str(row.get('Context', ''))
    i = safe_str(row.get('Intervention', ''))
    o = safe_str(row.get('Outcome', ''))
    #删除最后的括号和括号中的内容
    o = re.sub(r'\([^)]*\)', '', o).strip()

    return f"Context: {c}. Intervention: {i}. Outcome: {o}."

# ================= 3. 主程序 =================

def main():
    print("⏳ 正在加载语义模型 (BAAI/bge-large-en-v1.5)...")
    # 使用你本地的路径
    model = SentenceTransformer('/data_share_from_3090/wy_code/code/EE/NER/utd24/BAAI/bge-large-en-v1.5')

    try:
        llm_df = pd.read_csv(LLM_FILE)
        gt_df = pd.read_csv(GT_FILE)
    except Exception as e:
        print(f"❌ 读取 CSV 失败，请检查文件路径和名称: {e}")
        return

    gt_df = gt_df.dropna(subset=['Context', 'Intervention', 'Outcome', 'Title'], how='all')

    print("\n🚀 开始进行【基于题目的多维度语义匹配评估】...\n")

    # ----- 阶段 1：计算 Recall 和 细粒度相似度 (以 GT 为基准遍历) -----
    total_gt_propositions = len(gt_df)
    matched_gt_count = 0
    missing_papers = 0
    
    # 细粒度分数记录
    c_scores, i_scores, o_scores = [], [],[]

    for index, gt_row in gt_df.iterrows():
        gt_title = gt_row.get('Title', '')
        if pd.isna(gt_title) or str(gt_title).strip() == "":
            continue

        candidate_mask = llm_df['Title'].apply(lambda x: is_title_match(gt_title, x))
        llm_candidates = llm_df[candidate_mask]
        
        if llm_candidates.empty:
            missing_papers += 1
            continue
            
        gt_sentence = concat_cio(gt_row)
        gt_embedding = model.encode(gt_sentence, convert_to_tensor=True)
        
        best_similarity = 0.0
        best_llm_row = None
        
        for _, llm_row in llm_candidates.iterrows():
            llm_sentence = concat_cio(llm_row)
            llm_embedding = model.encode(llm_sentence, convert_to_tensor=True)
            cos_sim = util.cos_sim(gt_embedding, llm_embedding).item()
            
            if cos_sim > best_similarity:
                best_similarity = cos_sim
                best_llm_row = llm_row
                best_llm_sentence = llm_sentence
                
        if best_similarity >= SIMILARITY_THRESHOLD:
            matched_gt_count += 1
            # 计算细粒度得分
            c_sim = util.cos_sim(model.encode(safe_str(gt_row['Context'])), model.encode(safe_str(best_llm_row['Context']))).item()
            i_sim = util.cos_sim(model.encode(safe_str(gt_row['Intervention'])), model.encode(safe_str(best_llm_row['Intervention']))).item()
            o_sim = util.cos_sim(model.encode(safe_str(gt_row['Outcome'])), model.encode(safe_str(best_llm_row['Outcome']))).item()
            
            c_scores.append(c_sim)
            i_scores.append(i_sim)
            o_scores.append(o_sim)
            print(f"[+] 召回成功! (得分: {best_similarity:.2f}) | 论文: {gt_title[:40]}...")
            print(f"    👉 黄金标准: {gt_sentence}...")
            print(f"    🤖 LLM最佳: {best_llm_sentence}...")
        
        else:
            print(f"[x] 召回失败! (最高得分: {best_similarity:.2f}) | 论文: {gt_title[:40]}...")

    # ----- 阶段 2：计算 Precision 和 格式依从率 (以 LLM 提取结果为基准遍历) -----
    print("\n🚀 开始计算 精确率 (Precision) 与 格式规范性...")
    
    # 筛选出属于这批目标论文的 LLM 提取条目 (不惩罚大模型额外提取了其他无关论文)
    gt_titles_list = gt_df['Title'].dropna().unique()
    valid_llm_rows =[]
    
    format_compliant_count = 0 # 符合语法的条目数
    
    for _, llm_row in llm_df.iterrows():
        llm_title = llm_row.get('Title', '')
        if any(is_title_match(llm_title, t) for t in gt_titles_list):
            valid_llm_rows.append(llm_row)
            
            # 格式检查：Intervention 必须 -ing 开头，Outcome 必须 -s/-es 动词开头
            i_text = safe_str(llm_row.get('Intervention', '')).strip()
            o_text = safe_str(llm_row.get('Outcome', '')).strip()
            # 简单启发式正则：第一个词是 xxxing，且 outcome 首词是 xxxs 
            i_ok = bool(re.match(r'^[a-zA-Z]+ing\b', i_text, re.IGNORECASE))
            o_ok = bool(re.match(r'^[a-zA-Z]+s\b', o_text, re.IGNORECASE))
            if i_ok and o_ok:
                format_compliant_count += 1

    total_llm_extractions = len(valid_llm_rows)
    matched_llm_count = 0

    for llm_row in valid_llm_rows:
        llm_title = llm_row.get('Title', '')
        gt_candidates = gt_df[gt_df['Title'].apply(lambda x: is_title_match(llm_title, x))]
        
        llm_sentence = concat_cio(llm_row)
        llm_embedding = model.encode(llm_sentence, convert_to_tensor=True)
        
        best_sim = 0.0
        for _, gt_row in gt_candidates.iterrows():
            gt_sentence = concat_cio(gt_row)
            gt_embedding = model.encode(gt_sentence, convert_to_tensor=True)
            sim = util.cos_sim(llm_embedding, gt_embedding).item()
            if sim > best_sim:
                best_sim = sim
                
        if best_sim >= SIMILARITY_THRESHOLD:
            matched_llm_count += 1

    # ================= 4. 输出最终指标 =================
    if total_gt_propositions == 0 or total_llm_extractions == 0:
        print("\n❌ 错误：数据不足，无法计算指标。")
        return

    recall = matched_gt_count / total_gt_propositions
    precision = matched_llm_count / total_llm_extractions
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    format_rate = format_compliant_count / total_llm_extractions
    
    print("\n" + "="*55)
    print(f"📊 【大模型信息抽取多维综合评估报告】 📊")
    print("="*55)
    print(f"▶️ 模型路径: BAAI/bge-large-en-v1.5")
    print(f"▶️ 评估阈值 (Threshold): {SIMILARITY_THRESHOLD}")
    print(f"▶️ 黄金标准总条数 (GT): {total_gt_propositions}")
    print(f"▶️ LLM 针对黄金论文抽取的总条数: {total_llm_extractions}")
    print("-" * 55)
    print(f"🎯 语义召回率 (Recall):    {recall:.2%}  (还原了多少原论文命题)")
    print(f"🎯 语义精确率 (Precision): {precision:.2%}  (抽出来的内容有多少是对的)")
    print(f"🏆 综合 F1-Score:        {f1_score:.2%}  (抽取质量核心指标)")
    print("-" * 55)
    print(f"✍️ 格式依从率 (Instruction Following): {format_rate:.2%} (遵守-ing/-s语法约束的比例)")
    print("-" * 55)
    print("🧩 【成功匹配条目的细粒度要素相似度】")
    print(f"   - Context (情境) 平均相似度:      {np.mean(c_scores):.4f}" if c_scores else "   - Context: N/A")
    print(f"   - Intervention (干预) 平均相似度: {np.mean(i_scores):.4f}" if i_scores else "   - Intervention: N/A")
    print(f"   - Outcome (结果) 平均相似度:      {np.mean(o_scores):.4f}" if o_scores else "   - Outcome: N/A")
    print("="*55)

if __name__ == "__main__":
    main()