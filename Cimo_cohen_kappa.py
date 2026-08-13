"""
CIMO 人工一致性验证脚本 v1
==============================
覆盖三个维度：
  - Recall    : 0/1/2/3 有序评分  → 线性加权 Cohen's κ
  - Precision : TP/VN/FP 名义分类 → 标准 Cohen's κ
  - Theme     : 1-6 名义分类      → 标准 Cohen's κ（含分层一致率）

工作流：
  Step 1  分层抽样（Recall / Precision / Theme 各自独立抽样，精确凑满配额）
  Step 2  交互打分（三个维度依次进行，支持断点续跑 / 跳过 / 退出保存）
  Step 3  计算 Cohen's κ（含混淆矩阵、分维度分析、论文写作模板）

用法：
  python Cimo_cohen_kappa.py  --step 1
  python Cimo_human_kappa_v1.py --step 2 --dim recall
  python Cimo_human_kappa_v1.py --step 2 --dim precision
  python Cimo_human_kappa_v1.py --step 2 --dim theme
  python Cimo_human_kappa_v1.py --step 3
"""

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# 0. 配置
# ══════════════════════════════════════════════════════════════════

# 评测结果输入文件
RECALL_EVAL_CSV    = "./eval_recall_deepseek_v6.csv"    # 含 best_score / c_match / i_match / o_match
PRECISION_EVAL_CSV = "./eval_precision_deepseek_v6.csv" # 含 label (TP/VN/FP)
THEME_EVAL_CSV     = "./results_deepseek_v6.jsonl"      # 抽取结果（含 theme / theme_confidence）

# 抽样输出文件（Step 1 生成，Step 2 填写，Step 3 读取）
RECALL_SAMPLE_CSV    = "./human_recall_sample_v1.csv"
PRECISION_SAMPLE_CSV = "./human_precision_sample_v1.csv"
THEME_SAMPLE_CSV     = "./human_theme_sample_v1.csv"

# 汇总报告
KAPPA_REPORT_JSON = "./kappa_report_v1.json"

# 抽样数量（建议各 ≥ 50 条；若总量不足则取全部）
RECALL_SAMPLE_N    = 30
PRECISION_SAMPLE_N = 30
THEME_SAMPLE_N     = 30

RANDOM_SEED  = 42
JUDGE_MODEL  = "GLM-5"   # 用于论文写作模板，与主评测脚本一致

RECALL_LABELS    = [0, 1, 2, 3]
PRECISION_LABELS = ["TP", "VN", "FP"]
THEME_LABELS     = [1, 2, 3, 4, 5, 6]

THEME_NAMES = {
    1: "Governance mode & mechanism",
    2: "Network formation & initiation",
    3: "Interorganizational relationships",
    4: "Exploiting external resources",
    5: "Open innovation & learning",
    6: "Operational practices",
}


# ══════════════════════════════════════════════════════════════════
# 1. 通用工具
# ══════════════════════════════════════════════════════════════════

def _largest_remainder(sample_n: int, counts: dict, labels: list) -> dict:
    """
    最大余数法（Largest Remainder Method）：
    按各档比例精确分配 sample_n 个配额，余数部分按小数位从大到小补足。
    """
    total = sum(counts.values())
    if total == 0:
        return {l: 0 for l in labels}

    quotas    = {}
    remainders = {}
    for l in labels:
        exact        = sample_n * counts.get(l, 0) / total
        quotas[l]    = int(exact)
        remainders[l] = exact - int(exact)

    shortage = sample_n - sum(quotas.values())
    for l in sorted(remainders, key=remainders.get, reverse=True)[:shortage]:
        quotas[l] += 1

    return quotas


def _stratified_sample(df: pd.DataFrame, label_col: str,
                       labels: list, sample_n: int,
                       seed: int = RANDOM_SEED) -> pd.DataFrame:
    """分层抽样，精确凑满 sample_n（不超过各档实际数量）。"""
    counts  = {l: int((df[label_col] == l).sum()) for l in labels}
    quotas  = _largest_remainder(sample_n, counts, labels)

    print(f"\n  分层配额（精确凑满 {sample_n} 条）：")
    parts = []
    for l in labels:
        n   = min(quotas[l], counts[l])
        grp = df[df[label_col] == l]
        if n > 0 and len(grp) > 0:
            parts.append(grp.sample(n, random_state=seed))
        print(f"    {label_col}={l}: 共{counts[l]}条，配额{quotas[l]}，实际抽{n}条")

    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts)
              .sample(frac=1, random_state=seed)
              .reset_index(drop=True))


def _interpret_kappa(k: float) -> str:
    if k < 0.00:  return "< 0  一致性极差（低于随机水平）"
    if k < 0.20:  return "0.00-0.20  Slight（极低）"
    if k < 0.40:  return "0.20-0.40  Fair（较低）"
    if k < 0.60:  return "0.40-0.60  Moderate（中等）"
    if k < 0.80:  return "0.60-0.80  Substantial（较强，学术可接受）"
    return              "≥ 0.80  Almost perfect（近乎完全一致）"


def _weighted_kappa(y1: list, y2: list, labels: list) -> dict:
    """
    线性加权 Cohen's κ（适用于有序评分）。
    同时返回标准 κ 和加权 κ，以及混淆矩阵。
    """
    k         = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    conf      = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        if a in label_idx and b in label_idx:
            conf[label_idx[a]][label_idx[b]] += 1

    total    = conf.sum()
    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)

    po_exact = np.trace(conf) / total
    pe       = (row_sums @ col_sums) / (total ** 2)
    kappa    = (po_exact - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0

    max_diff = max(labels) - min(labels)
    weights  = np.array([
        [1 - abs(labels[i] - labels[j]) / max_diff for j in range(k)]
        for i in range(k)
    ])
    po_w  = (weights * conf).sum() / total
    pe_w  = (weights * np.outer(row_sums, col_sums)).sum() / (total ** 2)
    kappa_w = (po_w - pe_w) / (1 - pe_w) if (1 - pe_w) > 1e-9 else 0.0

    return {
        "exact_agree"    : round(float(po_exact), 4),
        "kappa"          : round(float(kappa), 4),
        "kappa_weighted" : round(float(kappa_w), 4),
        "conf"           : conf,
        "labels"         : labels,
    }


def _nominal_kappa(y1: list, y2: list, labels: list) -> dict:
    """
    标准 Cohen's κ（适用于名义分类：Precision TP/VN/FP、Theme 1-6）。
    """
    k         = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    conf      = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        if a in label_idx and b in label_idx:
            conf[label_idx[a]][label_idx[b]] += 1

    total    = conf.sum()
    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)

    po = np.trace(conf) / total
    pe = (row_sums @ col_sums) / (total ** 2)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else 0.0

    return {
        "exact_agree": round(float(po), 4),
        "kappa"      : round(float(kappa), 4),
        "conf"        : conf,
        "labels"     : labels,
    }


def _kappa_ci(kappa: float, n: int) -> tuple:
    """近似 95% 置信区间：SE ≈ sqrt((1 - κ²) / (n - 1))"""
    if n <= 1:
        return (float("nan"), float("nan"))
    se = np.sqrt((1 - kappa ** 2) / (n - 1))
    return (round(kappa - 1.96 * se, 4), round(kappa + 1.96 * se, 4))


def _print_conf_matrix(conf: np.ndarray, labels: list, row_name: str = "Judge",
                       col_name: str = "人工"):
    """动态生成混淆矩阵，不硬编码标签数量。"""
    header = f"  {'':8s}" + "".join(f"  {col_name}{l}" for l in labels)
    print(header)
    for i, l in enumerate(labels):
        row_str = f"  {row_name}{l}：".ljust(10) + "".join(
            f"{int(conf[i][j]):6d}" for j in range(len(labels))
        )
        print(row_str)


# ══════════════════════════════════════════════════════════════════
# 2. Recall — Step 1 抽样
# ══════════════════════════════════════════════════════════════════

def recall_step1():
    print("\n" + "=" * 62)
    print("  Recall 抽样（Step 1）")
    print("=" * 62)

    if not os.path.exists(RECALL_EVAL_CSV):
        print(f"[错误] 找不到评测文件：{RECALL_EVAL_CSV}")
        return

    df = pd.read_csv(RECALL_EVAL_CSV, encoding="utf-8-sig")

    # 过滤 C/I/O 三列均完整的条目
    required = ["best_pred_ctx", "best_pred_int", "best_pred_out"]
    missing  = [c for c in required if c not in df.columns]
    check    = required if not missing else ["best_pred_int"]
    if missing:
        print(f"[警告] 缺少列 {missing}，仅按 best_pred_int 过滤")

    mask = pd.Series([True] * len(df))
    for col in check:
        mask &= df[col].notna() & (df[col].astype(str).str.strip() != "") & \
                (df[col].astype(str).str.strip() != "nan")
    valid = df[mask].copy()
    valid["best_score"] = valid["best_score"].astype(int)

    print(f"  有效条目（C/I/O 均完整）：{len(valid)}")
    print("  Judge 分布：")
    print(valid["best_score"].value_counts().sort_index().to_string())

    sample = _stratified_sample(valid, "best_score", RECALL_LABELS, RECALL_SAMPLE_N)
    if sample.empty:
        print("[错误] 无有效样本")
        return

    # 添加人工标注列
    for col in ["human_score", "human_c", "human_i", "human_o", "human_note"]:
        sample[col] = ""

    keep = [c for c in [
        "gold_title", "gold_theme", "gold_context", "gold_interv", "gold_outcome",
        "best_pred_ctx", "best_pred_int", "best_pred_out",
        "best_score", "c_match", "i_match", "o_match", "best_reason",
        "human_score", "human_c", "human_i", "human_o", "human_note",
    ] if c in sample.columns]
    sample = sample[keep]
    sample.insert(0, "id", range(1, len(sample) + 1))
    sample.to_csv(RECALL_SAMPLE_CSV, index=False, encoding="utf-8-sig")

    print(f"\n  抽样完成！{len(sample)} 条 → {RECALL_SAMPLE_CSV}")
    print("\n  打分说明（human_score 列填 0/1/2/3）：")
    print("    3 = I✓ 且 C 或 O 至少一个✓")
    print("    2 = 仅 I✓，C/O 均不匹配")
    print("    1 = C 或 O 匹配，但 I 不匹配")
    print("    0 = 无任何维度匹配")
    print("  可选：同时填写 human_c/i/o（1=匹配，0=不匹配）以支持分维度分析")
    print("  推荐：用 Excel 打开填写后保存，再运行 step 2 或直接 step 3")


# ══════════════════════════════════════════════════════════════════
# 3. Precision — Step 1 抽样
# ══════════════════════════════════════════════════════════════════

def precision_step1():
    print("\n" + "=" * 62)
    print("  Precision 抽样（Step 1）")
    print("=" * 62)

    if not os.path.exists(PRECISION_EVAL_CSV):
        print(f"[错误] 找不到评测文件：{PRECISION_EVAL_CSV}")
        return

    df = pd.read_csv(PRECISION_EVAL_CSV, encoding="utf-8-sig")
    valid = df[df["label"].isin(PRECISION_LABELS)].copy()
    print(f"  有效条目（TP/VN/FP）：{len(valid)}")
    print("  Judge 分布：")
    print(valid["label"].value_counts().to_string())

    sample = _stratified_sample(valid, "label", PRECISION_LABELS, PRECISION_SAMPLE_N)
    if sample.empty:
        print("[错误] 无有效样本")
        return

    for col in ["human_label", "human_note"]:
        sample[col] = ""

    keep = [c for c in [
        "paper_title", "pred_context", "pred_interv", "pred_outcome",
        "gold_count", "label", "reason",
        "human_label", "human_note",
    ] if c in sample.columns]
    sample = sample[keep]
    sample.insert(0, "id", range(1, len(sample) + 1))
    sample.to_csv(PRECISION_SAMPLE_CSV, index=False, encoding="utf-8-sig")

    print(f"\n  抽样完成！{len(sample)} 条 → {PRECISION_SAMPLE_CSV}")
    print("\n  打分说明（human_label 列填 TP/VN/FP）：")
    print("    TP = 与参考 CIMO 三维度均匹配（抽象层次差异可接受）")
    print("    VN = 未在参考列表，但纸面有充分依据且 I 是明确行动")
    print("    FP = 不支持、I 是条件而非行动，或 C/I/O 不连贯")


# ══════════════════════════════════════════════════════════════════
# 4. Theme — Step 1 抽样
# ══════════════════════════════════════════════════════════════════

def theme_step1():
    print("\n" + "=" * 62)
    print("  Theme 抽样（Step 1）")
    print("=" * 62)

    if not os.path.exists(THEME_EVAL_CSV):
        print(f"[错误] 找不到抽取结果文件：{THEME_EVAL_CSV}")
        return

    rows = []
    with open(THEME_EVAL_CSV, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "_error" in rec:
                continue
            for item in rec.get("cimo_list", []):
                theme_str = item.get("theme", "")
                import re
                m = re.search(r"Theme\s*(\d)", str(theme_str), re.IGNORECASE)
                if not m:
                    continue
                t = int(m.group(1))
                if t not in THEME_LABELS:
                    continue
                rows.append({
                    "paper_title"       : rec.get("title", "")[:70],
                    "_folder"           : rec.get("_folder", ""),
                    "cimo_id"           : item.get("id", ""),
                    "context"           : item.get("context_abstract") or item.get("context", ""),
                    "intervention"      : item.get("intervention_abstract") or item.get("intervention", ""),
                    "outcome"           : item.get("outcome_abstract") or item.get("outcome", ""),
                    "model_theme"       : t,
                    "model_theme_str"   : theme_str[:60],
                    "theme_confidence"  : item.get("theme_confidence", ""),
                    "theme_alt"         : item.get("theme_alt", ""),
                    "human_theme"       : "",
                    "human_note"        : "",
                })

    if not rows:
        print("[错误] 未从 JSONL 中读取到有效 CIMO 条目")
        return

    df = pd.DataFrame(rows)
    print(f"  有效 CIMO 条目：{len(df)}")
    print("  模型 Theme 分布：")
    print(df["model_theme"].value_counts().sort_index().to_string())

    sample = _stratified_sample(df, "model_theme", THEME_LABELS, THEME_SAMPLE_N)
    if sample.empty:
        print("[错误] 无有效样本")
        return

    sample.insert(0, "id", range(1, len(sample) + 1))
    sample.to_csv(THEME_SAMPLE_CSV, index=False, encoding="utf-8-sig")

    print(f"\n  抽样完成！{len(sample)} 条 → {THEME_SAMPLE_CSV}")
    print("\n  打分说明（human_theme 列填 1-6）：")
    for t, name in THEME_NAMES.items():
        print(f"    {t} = {name}")
    print("\n  重点：基于 C/I/O 判断核心管理问题属于哪个主题，")
    print("         不要被 model_theme_str 的标签文字影响。")


# ══════════════════════════════════════════════════════════════════
# 5. 通用交互打分（Step 2）
# ══════════════════════════════════════════════════════════════════

def _interactive_score(sample_csv: str,
                       display_fn,
                       score_fn,
                       valid_inputs: list,
                       prompt_text: str):
    """
    通用交互打分框架。
      display_fn(row) → 打印当前条目的详情
      score_fn(df, idx, raw_input) → 将用户输入写入 df[idx]，返回 True/False
    """
    if not os.path.exists(sample_csv):
        print(f"[错误] 找不到抽样文件：{sample_csv}，请先运行 step 1")
        return

    df = pd.read_csv(sample_csv, encoding="utf-8-sig")

    # 判断哪一列是"已打分"的判断依据（取第一个 human_ 开头的列）
    human_cols = [c for c in df.columns if c.startswith("human_") and "note" not in c]
    if not human_cols:
        print("[错误] 样本文件中未找到 human_ 列")
        return
    score_col = human_cols[0]

    unscored = df[df[score_col].astype(str).str.strip().isin(["", "nan"])].index.tolist()

    if not unscored:
        print("所有条目已打分，请直接运行 step 3")
        return

    print("=" * 65)
    print(f"  人工标注（剩余 {len(unscored)} 条）  q=退出保存  s=跳过当前")
    print("=" * 65)
    print(f"  有效输入：{valid_inputs}")
    print()

    skipped = 0
    for pos, idx in enumerate(unscored):
        row = df.loc[idx]
        print(f"\n{'─' * 65}")
        print(f"  [{pos + 1}/{len(unscored)}] {str(row.get('paper_title', row.get('gold_title', '')))[:60]}")
        print(f"{'─' * 65}")
        display_fn(row)

        while True:
            raw = input(f"\n  {prompt_text} (q=退出, s=跳过): ").strip().lower()
            if raw == "q":
                df.to_csv(sample_csv, index=False, encoding="utf-8-sig")
                print(f"\n  已保存（{pos} 条已评，{skipped} 条跳过）→ {sample_csv}")
                return
            if raw == "s":
                skipped += 1
                break
            if score_fn(df, idx, raw):
                break
            print(f"  无效输入，请重新输入（有效值：{valid_inputs}）")

    df.to_csv(sample_csv, index=False, encoding="utf-8-sig")
    total_scored = df[score_col].astype(str).str.strip().isin(
        [str(v) for v in valid_inputs]
    ).sum()
    print(f"\n  打分完成！有效 {total_scored} 条，跳过 {skipped} 条 → {sample_csv}")
    print("  运行 step 3 计算 Cohen's κ")


# ── Recall 交互打分 ──────────────────────────────────────────────

def recall_step2():
    def display(row):
        print(f"\n  【黄金集】")
        print(f"    C: {row.get('gold_context', '')}")
        print(f"    I: {row.get('gold_interv', '')}")
        print(f"    O: {row.get('gold_outcome', '')}")
        print(f"\n  【模型输出】")
        print(f"    C: {row.get('best_pred_ctx', '')}")
        print(f"    I: {row.get('best_pred_int', '')}")
        print(f"    O: {row.get('best_pred_out', '')}")
        c = "✓" if str(row.get("c_match", "")).lower() == "true" else "✗"
        i = "✓" if str(row.get("i_match", "")).lower() == "true" else "✗"
        o = "✓" if str(row.get("o_match", "")).lower() == "true" else "✗"
        print(f"\n  Judge={row.get('best_score', '?')}/3  "
              f"(C{c} I{i} O{o})  理由：{str(row.get('best_reason', ''))[:70]}")
        print("\n  评分标准：3=I✓+(C或O)✓  2=仅I✓  1=C或O✓但I✗  0=全不匹配")

    def score_fn(df, idx, raw):
        if raw not in ("0", "1", "2", "3"):
            return False
        df.at[idx, "human_score"] = int(raw)
        print("  可选分维度（直接回车跳过）")
        hc = input("    C 匹配？(1/0): ").strip()
        hi = input("    I 匹配？(1/0): ").strip()
        ho = input("    O 匹配？(1/0): ").strip()
        note = input("    备注: ").strip()
        df.at[idx, "human_c"]    = hc if hc in ("0", "1") else ""
        df.at[idx, "human_i"]    = hi if hi in ("0", "1") else ""
        df.at[idx, "human_o"]    = ho if ho in ("0", "1") else ""
        df.at[idx, "human_note"] = note
        return True

    _interactive_score(
        RECALL_SAMPLE_CSV, display, score_fn,
        valid_inputs=["0", "1", "2", "3"],
        prompt_text="你的评分 (0/1/2/3)",
    )


# ── Precision 交互打分 ───────────────────────────────────────────

def precision_step2():
    def display(row):
        print(f"\n  【模型输出 CIMO】")
        print(f"    C: {row.get('pred_context', '')}")
        print(f"    I: {row.get('pred_interv', '')}")
        print(f"    O: {row.get('pred_outcome', '')}")
        print(f"\n  Judge 标签：{row.get('label', '?')}  "
              f"理由：{str(row.get('reason', ''))[:70]}")
        print("\n  分类标准：")
        print("    TP = 与黄金集三维度匹配（抽象层次差异可接受）")
        print("    VN = 纸面有依据、I 是明确行动、但不在黄金集中")
        print("    FP = 不支持 / I 是条件 / C-I-O 不连贯")

    def score_fn(df, idx, raw):
        label = raw.upper()
        if label not in ("TP", "VN", "FP"):
            return False
        df.at[idx, "human_label"] = label
        note = input("    备注（可选）: ").strip()
        df.at[idx, "human_note"] = note
        return True

    _interactive_score(
        PRECISION_SAMPLE_CSV, display, score_fn,
        valid_inputs=["tp", "vn", "fp"],
        prompt_text="你的标签 (TP/VN/FP)",
    )


# ── Theme 交互打分 ───────────────────────────────────────────────

def theme_step2():
    def display(row):
        print(f"\n  【CIMO】")
        print(f"    C: {row.get('context', '')}")
        print(f"    I: {row.get('intervention', '')}")
        print(f"    O: {row.get('outcome', '')}")
        print(f"\n  模型标注：Theme {row.get('model_theme', '?')} "
              f"（置信度={row.get('theme_confidence', '')}  "
              f"备选={row.get('theme_alt', '')}）")
        print(f"  模型原文：{str(row.get('model_theme_str', ''))[:60]}")
        print("\n  主题定义：")
        for t, name in THEME_NAMES.items():
            print(f"    {t} = {name}")

    def score_fn(df, idx, raw):
        if not raw.isdigit() or int(raw) not in THEME_LABELS:
            return False
        df.at[idx, "human_theme"] = int(raw)
        note = input("    备注（可选）: ").strip()
        df.at[idx, "human_note"] = note
        return True

    _interactive_score(
        THEME_SAMPLE_CSV, display, score_fn,
        valid_inputs=[str(t) for t in THEME_LABELS],
        prompt_text=f"你的主题编号 (1-6)",
    )


# ══════════════════════════════════════════════════════════════════
# 6. Step 3 — 计算 Cohen's κ（三个维度）
# ══════════════════════════════════════════════════════════════════

def _load_scored(sample_csv: str, judge_col: str, human_col: str,
                 valid_human: list, cast=None) -> tuple:
    """加载已打分样本，返回 (judge_list, human_list, df_scored, n_skipped)"""
    if not os.path.exists(sample_csv):
        return [], [], pd.DataFrame(), 0

    df = pd.read_csv(sample_csv, encoding="utf-8-sig")
    total = len(df)

    str_valid = [str(v) for v in valid_human]
    mask = df[human_col].astype(str).str.strip().isin(str_valid)
    scored = df[mask].copy()
    n_skip = total - len(scored)

    if n_skip > 0:
        print(f"  [注意] {total} 条中有 {n_skip} 条未打分（已排除）")

    if len(scored) == 0:
        return [], [], pd.DataFrame(), n_skip

    if cast:
        scored[judge_col]  = scored[judge_col].apply(cast)
        scored[human_col]  = scored[human_col].astype(str).str.strip().apply(cast)

    return (scored[judge_col].tolist(),
            scored[human_col].tolist(),
            scored, n_skip)


def step3_kappa():
    print("\n" + "=" * 62)
    print("  Cohen's κ 计算（Step 3）")
    print("=" * 62)

    all_results = {}

    # ── ① Recall ────────────────────────────────────────────────
    print("\n── Recall ──────────────────────────────────────────────")
    y_judge, y_human, df_r, n_skip_r = _load_scored(
        RECALL_SAMPLE_CSV, "best_score", "human_score",
        RECALL_LABELS, cast=int,
    )

    if y_judge:
        res = _weighted_kappa(y_judge, y_human, RECALL_LABELS)
        n   = len(y_judge)
        kw  = res["kappa_weighted"]
        lo, hi = _kappa_ci(kw, n)

        print(f"\n  标注条目数       : {n}")
        print(f"  完全一致率       : {res['exact_agree']:.4f}  "
              f"({int(res['exact_agree']*n)}/{n})")
        print(f"  Cohen's κ（标准） : {res['kappa']:.4f}")
        print(f"  Cohen's κ（加权） : {kw:.4f}  ← 主要报告指标")
        print(f"  95% CI           : [{lo:.4f}, {hi:.4f}]")
        print(f"  解读             : {_interpret_kappa(kw)}")

        print(f"\n  混淆矩阵（行=Judge，列=人工）：")
        _print_conf_matrix(res["conf"], RECALL_LABELS)

        # 得分分布对比
        print(f"\n  得分分布对比：")
        print(f"  {'分数':>5}  {'Judge':>8}  {'人工':>8}")
        for s in RECALL_LABELS:
            print(f"  {s:>5}  {y_judge.count(s):>8}  {y_human.count(s):>8}")

        # 分维度 C/I/O 一致率
        dim_agree = {}
        for dim in ("c", "i", "o"):
            jc = f"{dim}_match"
            hc = f"human_{dim}"
            if jc in df_r.columns and hc in df_r.columns:
                sub = df_r[df_r[hc].astype(str).str.strip().isin(["0", "1"])].copy()
                if len(sub) > 0:
                    sub[jc] = sub[jc].astype(str).str.lower() == "true"
                    sub[hc] = sub[hc].astype(str).str.strip() == "1"
                    rate = (sub[jc] == sub[hc]).mean()
                    dim_agree[dim.upper()] = round(rate, 4)

        if dim_agree:
            print(f"\n  分维度一致率（Judge vs 人工）：")
            for d, rate in dim_agree.items():
                bar = "█" * int(rate * 20)
                print(f"    {d}: {rate:.4f}  {bar}")
        else:
            print("\n  [提示] 未检测到 human_c/i/o 数据，可在 Step2 或 CSV 中手动填写")

        # 分歧分析
        disagree = df_r[df_r["best_score"] != df_r["human_score"]]
        if len(disagree) > 0:
            print(f"\n  分歧条目：{len(disagree)} 条")
            for _, row in disagree.iterrows():
                diff = int(row["best_score"]) - int(row["human_score"])
                direction = f"Judge高{abs(diff)}" if diff > 0 else f"人工高{abs(diff)}"
                print(f"    [{direction}] Judge={row['best_score']} 人工={row['human_score']} | "
                      f"I: {str(row.get('gold_interv',''))[:30]} "
                      f"→ {str(row.get('best_pred_int',''))[:30]}")

        all_results["recall"] = {
            "n": n, "n_skipped": n_skip_r,
            "exact_agree": res["exact_agree"],
            "kappa": res["kappa"],
            "kappa_weighted": kw,
            "ci_95": [lo, hi],
            "dim_agree": dim_agree,
            "interpretation": _interpret_kappa(kw),
        }
    else:
        print(f"  [跳过] {RECALL_SAMPLE_CSV} 无有效打分数据")

    # ── ② Precision ─────────────────────────────────────────────
    print("\n── Precision ───────────────────────────────────────────")
    y_judge_p, y_human_p, df_p, n_skip_p = _load_scored(
        PRECISION_SAMPLE_CSV, "label", "human_label",
        PRECISION_LABELS, cast=lambda x: str(x).upper().strip(),
    )

    if y_judge_p:
        res_p = _nominal_kappa(y_judge_p, y_human_p, PRECISION_LABELS)
        n_p   = len(y_judge_p)
        kp    = res_p["kappa"]
        lo_p, hi_p = _kappa_ci(kp, n_p)

        print(f"\n  标注条目数       : {n_p}")
        print(f"  完全一致率       : {res_p['exact_agree']:.4f}  "
              f"({int(res_p['exact_agree']*n_p)}/{n_p})")
        print(f"  Cohen's κ        : {kp:.4f}  ← 主要报告指标")
        print(f"  95% CI           : [{lo_p:.4f}, {hi_p:.4f}]")
        print(f"  解读             : {_interpret_kappa(kp)}")

        print(f"\n  混淆矩阵（行=Judge，列=人工）：")
        _print_conf_matrix(res_p["conf"], PRECISION_LABELS)

        # 分 label 一致率
        print(f"\n  各类一致率：")
        for i, lab in enumerate(PRECISION_LABELS):
            idx = [k for k, (a, b) in enumerate(zip(y_judge_p, y_human_p))
                   if a == lab or b == lab]
            if idx:
                rate = sum(y_judge_p[k] == y_human_p[k] for k in idx) / len(idx)
                print(f"    {lab}: {rate:.4f}  (涉及 {len(idx)} 条)")

        all_results["precision"] = {
            "n": n_p, "n_skipped": n_skip_p,
            "exact_agree": res_p["exact_agree"],
            "kappa": kp,
            "ci_95": [lo_p, hi_p],
            "interpretation": _interpret_kappa(kp),
        }
    else:
        print(f"  [跳过] {PRECISION_SAMPLE_CSV} 无有效打分数据")

    # ── ③ Theme ─────────────────────────────────────────────────
    print("\n── Theme ───────────────────────────────────────────────")
    y_judge_t, y_human_t, df_t, n_skip_t = _load_scored(
        THEME_SAMPLE_CSV, "model_theme", "human_theme",
        THEME_LABELS, cast=lambda x: int(float(str(x).strip())),
    )

    if y_judge_t:
        res_t = _nominal_kappa(y_judge_t, y_human_t, THEME_LABELS)
        n_t   = len(y_judge_t)
        kt    = res_t["kappa"]
        lo_t, hi_t = _kappa_ci(kt, n_t)

        print(f"\n  标注条目数       : {n_t}")
        print(f"  完全一致率       : {res_t['exact_agree']:.4f}  "
              f"({int(res_t['exact_agree']*n_t)}/{n_t})")
        print(f"  Cohen's κ        : {kt:.4f}  ← 主要报告指标")
        print(f"  95% CI           : [{lo_t:.4f}, {hi_t:.4f}]")
        print(f"  解读             : {_interpret_kappa(kt)}")

        print(f"\n  混淆矩阵（行=Judge/模型，列=人工）：")
        _print_conf_matrix(res_t["conf"], THEME_LABELS, row_name="Mdl", col_name="人工")

        # 分 Theme 一致率
        theme_agree = {}
        print(f"\n  分 Theme 一致率：")
        for i, t in enumerate(THEME_LABELS):
            idx = [k for k, (a, b) in enumerate(zip(y_judge_t, y_human_t))
                   if a == t or b == t]
            if len(idx) >= 3:
                rate = sum(y_judge_t[k] == y_human_t[k] for k in idx) / len(idx)
                theme_agree[t] = {"n": len(idx), "agree": round(rate, 4)}
                bar = "█" * int(rate * 20)
                print(f"    Theme {t} ({THEME_NAMES[t][:30]:<30}): "
                      f"n={len(idx):3d}  {rate:.4f}  {bar}")

        all_results["theme"] = {
            "n": n_t, "n_skipped": n_skip_t,
            "exact_agree": res_t["exact_agree"],
            "kappa": kt,
            "ci_95": [lo_t, hi_t],
            "theme_agree": theme_agree,
            "interpretation": _interpret_kappa(kt),
        }
    else:
        print(f"  [跳过] {THEME_SAMPLE_CSV} 无有效打分数据")

    # ── 汇总打印 ──────────────────────────────────────────────────
    if all_results:
        print("\n" + "=" * 62)
        print("  汇总")
        print("=" * 62)
        print(f"  κ 解读参考（Landis & Koch 1977）：")
        print(f"    <0.20 Slight  0.20-0.40 Fair  0.40-0.60 Moderate")
        print(f"    0.60-0.80 Substantial  ≥0.80 Almost perfect")
        print()
        for dim, r in all_results.items():
            k_key = "kappa_weighted" if "kappa_weighted" in r else "kappa"
            k_val = r[k_key]
            lo, hi = r["ci_95"]
            print(f"  {dim.capitalize():<12}: "
                  f"n={r['n']:3d}  κ={k_val:.4f}  "
                  f"95%CI[{lo:.4f},{hi:.4f}]  "
                  f"{r['interpretation'].split()[0]}")

    # ── 论文写作模板 ───────────────────────────────────────────────
    print(f"\n{'─' * 62}")
    print("  论文写作模板（从配置读取 Judge 模型名）：")
    print(f"{'─' * 62}")
    r_k = all_results.get("recall", {})
    p_k = all_results.get("precision", {})
    t_k = all_results.get("theme", {})
    print(
        f'  "To validate {JUDGE_MODEL} as the automated judge, we drew stratified\n'
        f'  samples of {r_k.get("n","?")}, {p_k.get("n","?")}, and {t_k.get("n","?")}'
        f" items for Recall, Precision, and Theme classification\n"
        f"  respectively, and manually annotated each item. The weighted\n"
        f"  Cohen's κ for Recall scoring was "
        f'{r_k.get("kappa_weighted","?"):.4f} '
        f'(95% CI {r_k.get("ci_95",["?","?"])[0]:.4f}–'
        f'{r_k.get("ci_95",["?","?"])[1]:.4f}),\n'
        f"  Cohen's κ for Precision classification was "
        f'{p_k.get("kappa","?"):.4f}, and\n'
        f"  Cohen's κ for Theme classification was "
        f'{t_k.get("kappa","?"):.4f}, indicating\n'
        f'  {r_k.get("interpretation","").split("（")[0].lower()} inter-rater reliability."'
    )

    # ── JSON 报告 ─────────────────────────────────────────────────
    report = {
        "date"        : datetime.now().strftime("%Y-%m-%d"),
        "judge_model" : JUDGE_MODEL,
        "dimensions"  : all_results,
    }
    with open(KAPPA_REPORT_JSON, "w", encoding="utf-8") as f:
        # 把 numpy array 转为 list
        def _default(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError
        json.dump(report, f, ensure_ascii=False, indent=2, default=_default)
    print(f"\n  完整报告已保存 → {KAPPA_REPORT_JSON}")


# ══════════════════════════════════════════════════════════════════
# 7. 主入口
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIMO 人工一致性验证 v1")
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3], required=True,
        help="1=抽样  2=交互打分  3=计算κ",
    )
    parser.add_argument(
        "--dim", type=str,
        choices=["recall", "precision", "theme", "all"],
        default="all",
        help="step1/2 时指定维度（默认 all）",
    )
    args = parser.parse_args()

    if args.step == 1:
        if args.dim in ("recall", "all"):
            recall_step1()
        if args.dim in ("precision", "all"):
            precision_step1()
        if args.dim in ("theme", "all"):
            theme_step1()

    elif args.step == 2:
        if args.dim == "all":
            print("[提示] Step2 请指定维度：--dim recall / precision / theme")
        elif args.dim == "recall":
            recall_step2()
        elif args.dim == "precision":
            precision_step2()
        elif args.dim == "theme":
            theme_step2()

    elif args.step == 3:
        step3_kappa()