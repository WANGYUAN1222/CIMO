"""
CIMO Ground-Truth Evaluation Script v1
════════════════════════════════════════════════════════════════
策略：以黄金数据集（GT）为基准，用 Gemini 作 Judge 评测模型抽取质量

核心指标（三层）：
  ① 字段级  — context / intervention / outcome 各自与 GT 的语义相似度 (0-3)
  ② 实例级  — Recall（GT 覆盖率）、Precision（抽取准确率）、F1
  ③ 文档级  — 汇总后对比 DeepSeek vs. Qwen3
════════════════════════════════════════════════════════════════
"""

import json
import time
import re
import pathlib
import pandas as pd
from collections import defaultdict
from difflib import SequenceMatcher
from google import genai

# ══════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════
GEMINI_API_KEY = "AIzaSyCPX-w8QJ-L3rkk3IGxfz9Zov8kKI4u3vk"
JUDGE_MODEL    = "gemini-2.5-flash-preview-04-17"

GT_CSV         = "./table4_ground_truth_item.csv"        # 黄金数据集路径
DEEPSEEK_JSONL = "./cimo_results/results_deepseek_v2.jsonl"
QWEN3_JSONL    = "./cimo_results/results_qwen3_v2.jsonl"
OUT_DIR        = pathlib.Path("./cimo_eval_gt")
OUT_DIR.mkdir(exist_ok=True)

client = genai.Client(api_key=GEMINI_API_KEY)

# ══════════════════════════════════════════════
# Judge Prompt：GT 对齐评测
# ══════════════════════════════════════════════
GT_JUDGE_SYSTEM = """You are an expert evaluator for CIMO (Context/Intervention/Outcome) extraction in supply chain and inter-firm relationship research.

Your task: Compare a model's extracted CIMO instances against a GROUND TRUTH set from the same paper.

━━━ SCORING RUBRICS ━━━

For each GT instance, find the best matching extracted instance and score:

[context_match] 0-3
  3: Semantically equivalent — same conditions captured
  2: Mostly correct — key condition present but some details missing or extra
  1: Partial — related but misses core condition or includes wrong elements
  0: No match found or completely wrong

[intervention_match] 0-3
  3: Semantically equivalent action captured correctly as -ing verb phrase
  2: Mostly correct action, minor wording differences
  1: Partial — correct domain but action mis-specified or malformed
  0: No match, wrong action, or condition stated as intervention

[outcome_match] 0-3
  3: Semantically equivalent result, correct direction
  2: Mostly correct — right direction, minor details off
  1: Partial — related outcome but direction unclear or key element missing
  0: No match or wrong direction

━━━ DOCUMENT-LEVEL ━━━
After scoring all GT instances:

[recall] float 0.0-1.0
  Fraction of GT instances that have a matched extracted instance (score ≥ 4 total across C+I+O)

[precision] float 0.0-1.0
  Fraction of extracted instances that correspond to a valid GT concept (not hallucinated)

━━━ OUTPUT FORMAT ━━━
Output ONLY valid JSON. No markdown, no explanation.

{
  "matched_instances": [
    {
      "gt_id": 1,
      "gt_intervention": "<GT intervention text>",
      "best_match_extracted_id": <int or null>,
      "context_match": <0-3>,
      "intervention_match": <0-3>,
      "outcome_match": <0-3>,
      "field_total": <sum 0-9>,
      "match_note": "<brief reason for score>"
    }
  ],
  "recall": <0.0-1.0>,
  "precision": <0.0-1.0>,
  "f1": <0.0-1.0>,
  "hallucinated_ids": [<list of extracted instance ids not matching any GT>],
  "overall_comment": "<1-2 sentences on overall quality vs GT>"
}"""

GT_JUDGE_USER = """Paper Title: {title}

━━━ GROUND TRUTH CIO instances (authoritative) ━━━
{gt_text}

━━━ MODEL EXTRACTED CIO instances (to evaluate) ━━━
{extracted_text}

Instructions:
- Match each GT instance to the best extracted instance.
- If no extracted instance matches a GT instance, scores are 0.
- If extracted has extra instances with no GT counterpart, mark them as hallucinated.
- Compute recall, precision, F1 based on matching threshold (field_total >= 4 out of 9 = matched).

Output JSON only."""


# ══════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════
def load_ground_truth(csv_path: str) -> dict:
    """加载 GT，按 Title 分组 → {title: [cio_dict, ...]}"""
    df = pd.read_csv(csv_path)
    gt = defaultdict(list)
    for _, row in df.iterrows():
        # 清理 Outcome 中的引用标注 "(Author, year)"
        outcome_clean = re.sub(r'\s*\(.*?\)\s*\.?\s*$', '', str(row['Outcome'])).strip()
        gt[str(row['Title']).strip()].append({
            'context':      str(row['Context']).strip(),
            'intervention': str(row['Intervention']).strip(),
            'outcome':      outcome_clean,
            'theme':        str(row['Research Theme']).strip(),
        })
    print(f"[GT] 加载完成：{len(gt)} 篇论文，{sum(len(v) for v in gt.values())} 条实例")
    return gt


def load_jsonl(path: str) -> dict:
    """加载模型抽取结果 → {_folder: record}"""
    records = {}
    p = pathlib.Path(path)
    if not p.exists():
        print(f"  [跳过] 文件不存在: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = r.get("_folder", "")
                if key and "_error" not in r:
                    records[key] = r
            except json.JSONDecodeError:
                continue
    print(f"[JSONL] {path}: {len(records)} 篇有效记录")
    return records


# ══════════════════════════════════════════════
# 标题模糊匹配
# ══════════════════════════════════════════════
def title_similarity(a: str, b: str) -> float:
    """基于字符序列的标题相似度"""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def match_title_to_gt(extracted_title: str, gt_titles: list,
                      threshold: float = 0.55) -> str | None:
    """将模型抽取的标题映射到最近的 GT 标题"""
    best_score = 0.0
    best_title = None
    for gt_title in gt_titles:
        score = title_similarity(extracted_title, gt_title)
        if score > best_score:
            best_score = score
            best_title = gt_title
    return best_title if best_score >= threshold else None


def build_title_index(records: dict, gt_titles: list) -> dict:
    """为每条抽取记录找到对应的 GT 标题 → {_folder: gt_title}"""
    index = {}
    unmatched = []
    for folder_key, record in records.items():
        extracted_title = record.get("title", "")
        gt_title = match_title_to_gt(extracted_title, gt_titles)
        if gt_title:
            index[folder_key] = gt_title
        else:
            unmatched.append((folder_key, extracted_title))
    if unmatched:
        print(f"  [警告] {len(unmatched)} 篇论文未能匹配到 GT 标题:")
        for k, t in unmatched[:5]:
            print(f"    {k[:40]} → '{t[:60]}'")
    return index


# ══════════════════════════════════════════════
# 格式化输出
# ══════════════════════════════════════════════
def format_gt_for_judge(gt_list: list) -> str:
    lines = []
    for i, item in enumerate(gt_list, 1):
        lines.append(f"GT Instance #{i}:")
        lines.append(f"  Context:      {item['context']}")
        lines.append(f"  Intervention: {item['intervention']}")
        lines.append(f"  Outcome:      {item['outcome']}")
        lines.append("")
    return "\n".join(lines)


def format_extracted_for_judge(cio_list: list) -> str:
    lines = []
    for item in cio_list:
        idx = item.get('id', '?')
        lines.append(f"Extracted Instance #{idx}:")
        lines.append(f"  Context:      {item.get('context', 'N/A')}")
        lines.append(f"  Intervention: {item.get('intervention', 'N/A')}")
        lines.append(f"  Outcome:      {item.get('outcome', 'N/A')}")
        lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════
# JSON 清理 & Gemini 调用
# ══════════════════════════════════════════════
def clean_json(raw: str) -> str:
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if '```' in raw:
        for part in raw.split('```'):
            part = part.strip()
            if part.startswith('json'):
                part = part[4:].strip()
            if part.startswith('{'):
                return part
    # 找到第一个 { ... } 块
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        return raw[start:end+1]
    return raw.strip()


def call_gemini(system_prompt: str, user_prompt: str,
                retries: int = 3, sleep_retry: float = 8.0) -> str:
    full_prompt = f"{system_prompt}\n\n{'─'*60}\n\n{user_prompt}"
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=full_prompt,
            )
            return response.text.strip()
        except Exception as e:
            print(f"    [Gemini重试 {attempt+1}/{retries}] {e}")
            time.sleep(sleep_retry * (attempt + 1))
    return ""


# ══════════════════════════════════════════════
# 核心评测函数
# ══════════════════════════════════════════════
def evaluate_one_paper(gt_list: list, cio_list: list,
                        title: str, model_name: str) -> dict | None:
    """用 Gemini Judge 对单篇论文进行 GT 对齐评测"""
    gt_text        = format_gt_for_judge(gt_list)
    extracted_text = format_extracted_for_judge(cio_list)

    raw = call_gemini(
        GT_JUDGE_SYSTEM,
        GT_JUDGE_USER.format(
            title=title,
            gt_text=gt_text,
            extracted_text=extracted_text,
        )
    )
    if not raw:
        return None
    try:
        result = json.loads(clean_json(raw))
        result["_title"]  = title
        result["_model"]  = model_name
        result["_n_gt"]   = len(gt_list)
        result["_n_extracted"] = len(cio_list)
        return result
    except Exception as e:
        print(f"    [解析失败] {e} | raw[:150]: {raw[:150]}")
        return None


# ══════════════════════════════════════════════
# 批量评测
# ══════════════════════════════════════════════
# 
#
def batch_evaluate(records: dict, gt_data: dict,
                   title_index: dict, model_name: str,
                   sleep_sec: float = 4.0, limit: int = None) -> list: # 1. 添加 limit 参数
    results = []
    # 找出所有既在模型结果中又在 GT 中的 key
    matchable = [(k, title_index[k]) for k in records if k in title_index]
    
    # 2. 如果设置了 limit，则只取前 N 条
    if limit is not None:
        matchable = matchable[:limit]
        print(f"\n[试运行] 模式开启：仅评测前 {limit} 篇论文")

    print(f"\n[评测] {model_name} — {len(matchable)} 篇待评测论文\n")


    for i, (folder_key, gt_title) in enumerate(matchable):
        record   = records[folder_key]
        gt_list  = gt_data[gt_title]
        cio_list = record.get("cio_list", [])

        print(f"  [{i+1}/{len(matchable)}] {folder_key[:50]}")
        print(f"    GT: {len(gt_list)} 条 | 抽取: {len(cio_list)} 条")

        if not cio_list:
            print(f"    [跳过] 无抽取结果")
            results.append({
                "_folder": folder_key, "_title": gt_title,
                "_model": model_name, "recall": 0.0,
                "precision": 0.0, "f1": 0.0,
                "_n_gt": len(gt_list), "_n_extracted": 0,
                "matched_instances": [],
                "_skip": "no_extraction"
            })
            continue

        eval_result = evaluate_one_paper(gt_list, cio_list, gt_title, model_name)

        if eval_result:
            eval_result["_folder"] = folder_key
            recall    = eval_result.get("recall", 0)
            precision = eval_result.get("precision", 0)
            f1        = eval_result.get("f1", 0)
            print(f"    Recall={recall:.2f} | Precision={precision:.2f} | F1={f1:.2f}")
        else:
            eval_result = {
                "_folder": folder_key, "_title": gt_title,
                "_model": model_name, "_parse_error": True,
                "recall": 0.0, "precision": 0.0, "f1": 0.0,
                "matched_instances": [],
            }
            print(f"    [评测失败]")

        results.append(eval_result)
        time.sleep(sleep_sec)

    return results




# ══════════════════════════════════════════════
# 统计汇总
# ══════════════════════════════════════════════
def compute_field_averages(results: list) -> dict:
    """计算 context / intervention / outcome 平均得分"""
    totals = defaultdict(list)
    for r in results:
        for inst in r.get("matched_instances", []):
            for field in ("context_match", "intervention_match", "outcome_match", "field_total"):
                if field in inst:
                    totals[field].append(inst[field])
    return {k: sum(v)/len(v) if v else 0.0 for k, v in totals.items()}


def summarize(results: list, model_name: str):
    valid = [r for r in results if "recall" in r and not r.get("_skip")]
    if not valid:
        print(f"[{model_name}] 无有效结果")
        return {}

    recalls    = [r["recall"]    for r in valid]
    precisions = [r["precision"] for r in valid]
    f1s        = [r["f1"]        for r in valid]

    avg_recall    = sum(recalls)    / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_f1        = sum(f1s)        / len(f1s)

    field_avgs = compute_field_averages(valid)

    print(f"\n{'═'*60}")
    print(f"  汇总报告 — {model_name}  (n={len(valid)} 篇)")
    print(f"{'═'*60}")
    print(f"  {'Recall (GT覆盖率)':<30} {avg_recall:.3f}")
    print(f"  {'Precision (抽取准确率)':<30} {avg_precision:.3f}")
    print(f"  {'F1':<30} {avg_f1:.3f}")
    print(f"  {'─'*50}")
    print(f"  字段级平均得分 (满分3)：")
    print(f"  {'  Context  Match':<30} {field_avgs.get('context_match',0):.2f}/3")
    print(f"  {'  Intervention Match':<30} {field_avgs.get('intervention_match',0):.2f}/3")
    print(f"  {'  Outcome Match':<30} {field_avgs.get('outcome_match',0):.2f}/3")
    print(f"  {'  Per-Instance Total':<30} {field_avgs.get('field_total',0):.2f}/9")
    print(f"{'═'*60}\n")

    return {
        "model": model_name, "n_papers": len(valid),
        "avg_recall": avg_recall, "avg_precision": avg_precision, "avg_f1": avg_f1,
        "field_avgs": field_avgs,
    }


# ══════════════════════════════════════════════
# 对比报告
# ══════════════════════════════════════════════
def compare_models(summary_a: dict, summary_b: dict):
    print(f"\n{'╔'+'═'*56+'╗'}")
    print(f"  🏆 模型对比报告")
    print(f"{'╠'+'═'*56+'╣'}")

    metrics = [
        ("Recall",    "avg_recall"),
        ("Precision", "avg_precision"),
        ("F1",        "avg_f1"),
    ]
    field_metrics = [
        ("Context Match",      "context_match"),
        ("Intervention Match", "intervention_match"),
        ("Outcome Match",      "outcome_match"),
    ]

    winners = {summary_a["model"]: 0, summary_b["model"]: 0}

    for label, key in metrics:
        va = summary_a.get(key, 0)
        vb = summary_b.get(key, 0)
        winner = summary_a["model"] if va > vb else (summary_b["model"] if vb > va else "tie")
        if winner != "tie":
            winners[winner] += 1
        star_a = " ✓" if winner == summary_a["model"] else ""
        star_b = " ✓" if winner == summary_b["model"] else ""
        print(f"  {label:<18} {summary_a['model']:>10}={va:.3f}{star_a:<3}  "
              f"{summary_b['model']:>8}={vb:.3f}{star_b}")

    print(f"  {'─'*54}")
    for label, key in field_metrics:
        va = summary_a["field_avgs"].get(key, 0)
        vb = summary_b["field_avgs"].get(key, 0)
        winner = summary_a["model"] if va > vb else (summary_b["model"] if vb > va else "tie")
        if winner != "tie":
            winners[winner] += 1
        star_a = " ✓" if winner == summary_a["model"] else ""
        star_b = " ✓" if winner == summary_b["model"] else ""
        print(f"  {label:<18} {summary_a['model']:>10}={va:.2f}{star_a:<3}  "
              f"{summary_b['model']:>8}={vb:.2f}{star_b}")

    final = max(winners, key=winners.get)
    print(f"  {'─'*54}")
    print(f"  综合胜出: {final}  "
          f"({summary_a['model']}={winners[summary_a['model']]} | "
          f"{summary_b['model']}={winners[summary_b['model']]})")
    print(f"{'╚'+'═'*56+'╝'}\n")


# ══════════════════════════════════════════════
# 保存 & 主入口
# ══════════════════════════════════════════════
def save_results(results: list, path: str):
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  → 已保存: {path}")


def main():
    # ── 1. 加载 GT ──
    gt_data   = load_ground_truth(GT_CSV)
    gt_titles = list(gt_data.keys())

    # ── 2. 加载模型结果 ──
    records_ds = load_jsonl(DEEPSEEK_JSONL)
    records_q3 = load_jsonl(QWEN3_JSONL)

    # 取前5条数据
    

    # ── 3. 标题匹配 ──
    print("\n[标题匹配] DeepSeek")
    idx_ds = build_title_index(records_ds, gt_titles)
    print(f"  匹配成功: {len(idx_ds)} / {len(records_ds)}")

    has_qwen3 = bool(records_q3)
    if has_qwen3:
        print("\n[标题匹配] Qwen3")
        idx_q3 = build_title_index(records_q3, gt_titles)
        print(f"  匹配成功: {len(idx_q3)} / {len(records_q3)}")

    # ── 4. GT 对齐评测 ──
    print("\n" + "═"*60)
    print("  开始 GT 对齐评测（DeepSeek）")
    print("═"*60)
    # 传入 limit=5
    eval_ds = batch_evaluate(records_ds, gt_data, idx_ds, "deepseek", limit=5) 
    save_results(eval_ds, str(OUT_DIR / "eval_deepseek_gt.jsonl"))

    summary_ds = summarize(eval_ds, "deepseek")

    if has_qwen3:
        print("\n" + "═"*60)
        print("  开始 GT 对齐评测（Qwen3）")
        print("═"*60)
        # 传入 limit=5
        eval_q3 = batch_evaluate(records_q3, gt_data, idx_q3, "qwen3", limit=5)
        save_results(eval_q3, str(OUT_DIR / "eval_qwen3_gt.jsonl"))

        summary_q3 = summarize(eval_q3, "qwen3")

        # ── 5. 对比报告 ──
        compare_models(summary_ds, summary_q3)

        # 保存汇总
        summary_path = str(OUT_DIR / "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"deepseek": summary_ds, "qwen3": summary_q3},
                      f, ensure_ascii=False, indent=2)
        print(f"  汇总已保存: {summary_path}")
    else:
        print("\n[提示] 未找到 Qwen3 结果文件，仅输出 DeepSeek 评测报告")
        summary_path = str(OUT_DIR / "summary_deepseek.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"deepseek": summary_ds}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()