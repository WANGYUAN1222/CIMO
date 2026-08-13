"""
Baseline 模型
=============
Baseline 1: Rule-based keyword extraction（主要 baseline）
  - 用预定义 CIMO 关键词从论文原文提取候选句子
  - 代表"传统 NLP / 无 LLM"方法
  
Baseline 2: Random baseline（统计下界）
  - 从黄金集随机采样作为预测
  - 理论 Recall 约等于随机概率，用于证明 LLM 结果有意义
"""

import json, re, random, pathlib
import pandas as pd
from difflib import SequenceMatcher

RANDOM_SEED   = 42
PAPER_FOLDER  = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/Golden_Paper"
GOLD_CSV      = "./table4_ground_truth_item.csv"
MAX_CHARS     = 40000   # 论文截断长度
AVG_CIMO_PER_PAPER = 5  # Random baseline 每篇采样数量（与 LLM 输出均值对齐）


# ══════════════════════════════════════════════
# CIMO 关键词词典（Rule-based baseline 核心）
# ══════════════════════════════════════════════
CONTEXT_PATTERNS = [
    r'\bin the context of\b', r'\bunder conditions? of\b',
    r'\bwhen\b.{5,40}\b(high|low|strong|weak|uncertain)',
    r'\bin\b.{3,30}\b(relationships?|alliances?|networks?|supply chain)',
    r'\bfor firms? (with|facing|in)\b',
    r'\bin (buyer.supplier|inter.firm|inter.organizational)',
]

INTERVENTION_PATTERNS = [
    r'\b(choosing|selecting|adopting|implementing|using|employing|investing in)\b',
    r'\b(building|developing|establishing|creating|forming)\b.{5,50}(ties|relationships?|alliances?)',
    r'\b(increasing|enhancing|improving|strengthening)\b.{5,40}(trust|commitment|integration)',
    r'\b(involving|engaging)\b.{5,40}(suppliers?|customers?|partners?)',
    r'\b(sharing|exchanging)\b.{5,30}(information|knowledge)',
    r'\b(formal contracts?|relational governance|hierarchical governance|market governance)',
]

OUTCOME_PATTERNS = [
    r'\b(leads? to|results? in|improves?|increases?|decreases?|reduces?|enhances?)\b.{5,60}(performance|satisfaction|trust|commitment|innovation|cost)',
    r'\b(positive(ly)?|negative(ly)?|significantly?)\b.{5,40}(effect|impact|influence|relationship)',
    r'\b(higher|lower|better|improved|reduced)\b.{5,40}(performance|outcome|result)',
]


def find_md(folder_name: str) -> str:
    if not folder_name: return ""
    root = pathlib.Path(PAPER_FOLDER)
    parts = folder_name.replace('\\', '/').split('/')
    for depth in range(len(parts), 0, -1):
        candidate = root.joinpath(*parts[:depth])
        if candidate.exists():
            mds = list(candidate.rglob("*.md"))
            if mds:
                try: return mds[0].read_text(encoding='utf-8')[:MAX_CHARS]
                except: return ""
    return ""


def extract_candidate_sentences(text: str, patterns: list) -> list:
    """从文本中提取匹配 pattern 的句子"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    results = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 300:
            continue
        for pat in patterns:
            if re.search(pat, sent, re.IGNORECASE):
                results.append(sent)
                break
    return results[:5]  # 每个维度最多取5个候选


def rule_based_extract(paper_text: str, title: str) -> list:
    """
    Rule-based CIMO 抽取
    策略：分别用关键词找 C/I/O 的候选句，然后组合成 CIMO 条目
    """
    if not paper_text:
        return []

    contexts      = extract_candidate_sentences(paper_text, CONTEXT_PATTERNS)
    interventions = extract_candidate_sentences(paper_text, INTERVENTION_PATTERNS)
    outcomes      = extract_candidate_sentences(paper_text, OUTCOME_PATTERNS)

    # 取每个维度最多 3 个，组合成 CIMO
    n = min(len(contexts), len(interventions), len(outcomes), 3)
    if n == 0:
        # 降级：至少用找到的句子填充
        n = max(1, min(
            max(len(contexts),1),
            max(len(interventions),1),
            max(len(outcomes),1)
        ))

    cimos = []
    for i in range(n):
        cimos.append({
            "id"          : i + 1,
            "context"     : contexts[i]      if i < len(contexts)      else "",
            "intervention": interventions[i]  if i < len(interventions) else "",
            "mechanism"   : None,
            "outcome"     : outcomes[i]       if i < len(outcomes)      else "",
        })
    return cimos


# ══════════════════════════════════════════════
# Baseline 1: Rule-based
# ══════════════════════════════════════════════
def run_rule_based(paper_folder_map: dict, output_jsonl: str):
    """
    paper_folder_map: {norm_title: folder_name}
    """
    print(f"Rule-based baseline | 共 {len(paper_folder_map)} 篇")
    results = []

    for i, (title, folder) in enumerate(paper_folder_map.items()):
        text   = find_md(folder)
        cimos  = rule_based_extract(text, title)
        print(f"  [{i+1}/{len(paper_folder_map)}] {title[:50]} → {len(cimos)} 条")
        results.append({
            "title"     : title,
            "cimo_list" : cimos,
            "_folder"   : folder,
            "_model"    : "rule_based",
        })

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"完成 → {output_jsonl}\n")


# ══════════════════════════════════════════════
# Baseline 2: Random
# ══════════════════════════════════════════════
def run_random_baseline(gold: dict, output_jsonl: str, n_per_paper: int = AVG_CIMO_PER_PAPER):
    """
    从黄金集所有条目里随机采样，模拟"随机预测"
    """
    random.seed(RANDOM_SEED)
    all_gold_items = [item for items in gold.values() for item in items]

    print(f"Random baseline | {len(gold)} 篇，每篇采样 {n_per_paper} 条")
    results = []

    for title in gold.keys():
        sampled = random.choices(all_gold_items, k=n_per_paper)
        cimos = [
            {
                "id"          : j + 1,
                "context"     : item.get('Context', ''),
                "intervention": item.get('Intervention', ''),
                "mechanism"   : None,
                "outcome"     : item.get('Outcome', ''),
            }
            for j, item in enumerate(sampled)
        ]
        results.append({
            "title"    : title,
            "cimo_list": cimos,
            "_folder"  : "",
            "_model"   : "random",
        })

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"完成 → {output_jsonl}\n")


# ══════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════
def norm(t):
    if not isinstance(t, str): return ''
    t = t.upper().strip()
    t = re.sub(r"[''`\u2018\u2019]", "'", t)
    t = re.sub(r'[\"\u201c\u201d]', '"', t)
    return re.sub(r'\s+', ' ', t)

def load_gold(path):
    df = pd.read_csv(path, encoding='utf-8')
    df['_n'] = df['Title'].apply(norm)
    return {n: grp.to_dict('records') for n, grp in df.groupby('_n')}

def build_folder_map(gold: dict, qwen_jsonl: str) -> dict:
    """从已有的 Qwen 结果里获取 title→folder 映射"""
    folder_map = {}
    with open(qwen_jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if '_error' in r: continue
            t = norm(r.get('title', ''))
            folder_map[t] = r.get('_folder', '')

    # 对黄金集里的 title 做模糊匹配
    pred_titles = list(folder_map.keys())
    result = {}
    for gt in gold.keys():
        if gt in folder_map:
            result[gt] = folder_map[gt]
        else:
            best = max(pred_titles,
                       key=lambda p: SequenceMatcher(None, gt, p).ratio(),
                       default=None)
            if best and SequenceMatcher(None, gt, best).ratio() >= 0.6:
                result[gt] = folder_map[best]
    return result


# ══════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════
if __name__ == "__main__":
    gold = load_gold(GOLD_CSV)

    # 从已有的 Qwen v3 结果获取 folder 映射（用于找论文 md 文件）
    QWEN_JSONL = "./results_qwen3_v3.jsonl"
    folder_map = build_folder_map(gold, QWEN_JSONL)
    print(f"folder 映射成功: {len(folder_map)}/{len(gold)} 篇\n")

    # Baseline 1: Rule-based
    run_rule_based(folder_map, "./results_rule_based.jsonl")

    # Baseline 2: Random
    run_random_baseline(gold, "./results_random.jsonl", n_per_paper=5)

    print("两个 baseline 生成完成！")
    print("接下来用 evaluate_equal.py 评测：")
    print("  在 EVAL_TARGETS 里加入：")
    print('  {"name":"Rule-based","jsonl":"./results_rule_based.jsonl",...}')
    print('  {"name":"Random",    "jsonl":"./results_random.jsonl",    ...}')