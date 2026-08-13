import json
import os
import time
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR  = "/data_share_from_3090/wy_code/code/EE/NER/utd24/gap_results_cimo"
RAG_DIR     = "/data_share_from_3090/wy_code/code/EE/NER/utd24/gap_results_rag"
EVAL_DIR    = "/data_share_from_3090/wy_code/code/EE/NER/utd24/gap_evaluation"
Path(EVAL_DIR).mkdir(exist_ok=True)

client = OpenAI(
    api_key  = "None",
    base_url = "http://172.17.65.42:8001/v1",
)
JUDGE_MODEL = "Qwen3-Next-80B-A3B-Instruct"

# ============================================================
# Step 1: 加载两组Gap结果
# ============================================================
def load_gaps(cimo_path, rag_path):
    with open(cimo_path, encoding="utf-8") as f:
        cimo_raw = json.load(f)
    with open(rag_path, encoding="utf-8") as f:
        rag_raw = json.load(f)

    def flatten(raw, method_label):
        rows = []
        for discipline, gaps in raw.items():
            for g in gaps:
                rows.append({
                    "method":                method_label,
                    "discipline":            discipline,
                    "dimension":             g.get("dimension", ""),
                    "gap_type":              g.get("gap_type", ""),
                    "gap_statement":         g.get("gap_statement", ""),
                    "evidence_pattern":      g.get("evidence_pattern", ""),
                    "related_tanskanen_gap": g.get("related_tanskanen_gap", ""),
                    "candidate_proposition": g.get("candidate_proposition", ""),
                })
        return rows

    cimo_rows = flatten(cimo_raw, "CIMO")
    rag_rows  = flatten(rag_raw,  "RAG")

    df = pd.DataFrame(cimo_rows + rag_rows)
    df = df[df["dimension"].isin(["C", "I", "M", "O"])].reset_index(drop=True)
    df["gap_id"] = df.index

    print(f"✓ 加载完成 — CIMO: {len(cimo_rows)}条  RAG: {len(rag_rows)}条")
    return df


# ============================================================
# Step 2: LLM自动评分（核心评估）
# ============================================================
JUDGE_PROMPT = """
You are an expert reviewer of Evidence-Based Management (EBM) research gaps.

Evaluate the following research gap on four dimensions.
Score each dimension from 1 to 3 (integer only):

Scoring rubric:
[Specificity]
  3 = Precisely identifies what is missing, in which context, and for whom
  2 = Identifies what is missing but context is vague
  1 = Too generic to guide research (e.g., "more research is needed")

[Novelty]
  3 = Points to a genuinely unexplored direction (absent evidence); pursuing it would yield a non-routine, original contribution
  2 = Extends an existing line of inquiry incrementally; only partially new
  1 = Restates an already well-addressed question; no new direction

[Actionability]
  3 = Directly translates to a testable research design or proposition
  2 = Suggests a direction but lacks concrete operationalization
  1 = Cannot be directly acted upon by researchers

[Theoretical_Grounding]
  3 = Supported by an explicit theoretical mechanism linking intervention to outcome (a clear generative "why")
  2 = Gestures at a theoretical basis, but the mechanism is implicit or underspecified
  1 = Atheoretical; an empirical absence with no generative explanation

Gap to evaluate:
- Dimension: {dimension}
- Gap Statement: {gap_statement}
- Evidence Pattern: {evidence_pattern}
- Candidate Proposition: {candidate_proposition}

Return ONLY valid JSON, no markdown:
{{
  "specificity":           <1|2|3>,
  "novelty":               <1|2|3>,
  "actionability":         <1|2|3>,
  "theoretical_grounding": <1|2|3>,
  "total":                 <sum of above>,
  "comment":               "<one sentence explaining the main strength or weakness>"
}}
"""

def judge_gap(row, max_retries=2):
    prompt = JUDGE_PROMPT.format(
        dimension             = row["dimension"],
        gap_statement         = str(row["gap_statement"])[:300],
        evidence_pattern      = str(row["evidence_pattern"])[:300],
        candidate_proposition = str(row["candidate_proposition"])[:300],
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model       = JUDGE_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 300,
                temperature = 0.0,
            )
            raw = response.choices[0].message.content.strip()

            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.split("```")[0].strip()

            scores = json.loads(raw)
            # 验证字段完整
            for key in ["specificity", "evidence_quality",
                        "actionability", "cimo_alignment"]:
                assert key in scores, f"缺少字段: {key}"
                assert scores[key] in [1, 2, 3], f"非法分值: {scores[key]}"

            return scores

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2)

    # 失败返回None
    return None


def run_judge_evaluation(df):
    print(f"\n开始LLM评分，共 {len(df)} 条gap...")
    results = []

    for idx, row in df.iterrows():
        scores = judge_gap(row)
        if scores:
            results.append({
                "gap_id":          row["gap_id"],
                "method":          row["method"],
                "discipline":      row["discipline"],
                "dimension":       row["dimension"],
                "gap_type":        row["gap_type"],
                "specificity":     scores["specificity"],
                "evidence_quality":scores["evidence_quality"],
                "actionability":   scores["actionability"],
                "cimo_alignment":  scores["cimo_alignment"],
                "total":           scores["total"],
                "comment":         scores.get("comment", ""),
                "gap_statement":   row["gap_statement"],
                "evidence_pattern": row["evidence_pattern"],
                "candidate_proposition": row["candidate_proposition"],
            })
        else:
            print(f"  ⚠ gap_id={row['gap_id']} 评分失败，跳过")

        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx+1}/{len(df)}")
        time.sleep(0.5)

    eval_df = pd.DataFrame(results)
    eval_df.to_excel(f"{EVAL_DIR}/judge_scores.xlsx", index=False)
    print(f"✓ 评分完成，有效结果 {len(eval_df)} 条")
    return eval_df


# ============================================================
# Step 3: 统计检验
# ============================================================
METRICS = ["specificity", "evidence_quality", "actionability",
           "cimo_alignment", "total"]

def run_statistical_tests(eval_df):
    print("\n" + "="*55)
    print("统计检验结果")
    print("="*55)

    cimo_df = eval_df[eval_df["method"] == "CIMO"]
    rag_df  = eval_df[eval_df["method"] == "RAG"]

    stat_rows = []

    for metric in METRICS:
        cimo_vals = cimo_df[metric].dropna().values
        rag_vals  = rag_df[metric].dropna().values

        cimo_mean = cimo_vals.mean()
        rag_mean  = rag_vals.mean()
        cimo_std  = cimo_vals.std()
        rag_std   = rag_vals.std()

        # Mann-Whitney U检验（不假设正态分布，适合1-3分的离散数据）
        u_stat, p_mw = stats.mannwhitneyu(
            cimo_vals, rag_vals, alternative="greater"
        )

        # Cohen's d效应量
        pooled_std = np.sqrt((cimo_std**2 + rag_std**2) / 2)
        cohens_d   = (cimo_mean - rag_mean) / (pooled_std + 1e-9)

        sig = "***" if p_mw < 0.001 else ("**" if p_mw < 0.01
              else ("*" if p_mw < 0.05 else "ns"))

        print(f"\n[{metric}]")
        print(f"  CIMO: {cimo_mean:.3f} ± {cimo_std:.3f}  (n={len(cimo_vals)})")
        print(f"  RAG : {rag_mean:.3f} ± {rag_std:.3f}  (n={len(rag_vals)})")
        print(f"  Mann-Whitney U={u_stat:.1f}, p={p_mw:.4f} {sig}")
        print(f"  Cohen's d={cohens_d:.3f}")

        stat_rows.append({
            "Metric":         metric,
            "CIMO_mean":      round(cimo_mean, 3),
            "CIMO_std":       round(cimo_std, 3),
            "RAG_mean":       round(rag_mean, 3),
            "RAG_std":        round(rag_std, 3),
            "CIMO_n":         len(cimo_vals),
            "RAG_n":          len(rag_vals),
            "U_statistic":    round(u_stat, 1),
            "p_value":        round(p_mw, 4),
            "significance":   sig,
            "Cohens_d":       round(cohens_d, 3),
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_excel(f"{EVAL_DIR}/statistical_tests.xlsx", index=False)
    print(f"\n✓ 统计结果已保存")
    return stat_df


# ============================================================
# Step 4: Cohen's κ 人机一致性验证
# （随机抽30条，人工打分后计算κ）
# ============================================================
def export_human_validation_sample(eval_df, n=30, random_state=42):
    """导出30条样本供人工打分，格式与LLM评分表一致"""
    sample = eval_df.sample(min(n, len(eval_df)), random_state=random_state)

    export_cols = ["gap_id", "method", "discipline", "dimension",
                   "gap_statement", "evidence_pattern",
                   "candidate_proposition",
                   "specificity", "evidence_quality",
                   "actionability", "cimo_alignment", "total"]

    sample_out = sample[export_cols].copy()

    # 新增空白列供人工填写
    for col in ["human_specificity", "human_evidence_quality",
                "human_actionability", "human_cimo_alignment"]:
        sample_out[col] = ""

    sample_out.to_excel(
        f"{EVAL_DIR}/human_validation_sample.xlsx", index=False
    )
    print(f"✓ 人工验证样本已导出: {EVAL_DIR}/human_validation_sample.xlsx")
    print("  请在 human_* 列填入你的评分(1/2/3)后，运行 compute_kappa()")
    return sample_out


def compute_kappa(human_file):
    """
    读取人工填写的验证表，计算加权Cohen's κ
    """
    from sklearn.metrics import cohen_kappa_score

    df = pd.read_excel(human_file)
    dims = ["specificity", "evidence_quality", "actionability", "cimo_alignment"]

    print("\n=== Cohen's κ 人机一致性 ===")
    kappas = {}
    for dim in dims:
        machine = df[dim].dropna().astype(int)
        human   = df[f"human_{dim}"].dropna().astype(int)

        # 取交集（两列都有值的行）
        valid = df[[dim, f"human_{dim}"]].dropna()
        if len(valid) < 5:
            print(f"  {dim}: 数据不足，跳过")
            continue

        k = cohen_kappa_score(
            valid[dim].astype(int),
            valid[f"human_{dim}"].astype(int),
            weights="linear"
        )
        kappas[dim] = round(k, 3)
        level = ("优秀(≥0.8)" if k >= 0.8 else
                 "良好(0.6-0.8)" if k >= 0.6 else
                 "一般(<0.6)")
        print(f"  {dim}: κ={k:.3f}  {level}")

    mean_k = np.mean(list(kappas.values()))
    print(f"\n  平均κ = {mean_k:.3f}")
    return kappas


# ============================================================
# Step 5: 可视化
# ============================================================
def plot_comparison(eval_df, stat_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("CIMO vs RAG: Research Gap Quality Comparison",
                 fontsize=14, fontweight="bold")

    cimo_color = "#2196F3"
    rag_color  = "#FF5722"

    # --- 图1: 各指标均值对比（条形图）---
    ax = axes[0]
    metrics_display = {
        "specificity":      "Specificity",
        "evidence_quality": "Evidence\nQuality",
        "actionability":    "Actionability",
        "cimo_alignment":   "CIMO\nAlignment",
        "total":            "Total",
    }
    x      = np.arange(len(metrics_display))
    width  = 0.35

    cimo_means = stat_df["CIMO_mean"].values
    rag_means  = stat_df["RAG_mean"].values
    cimo_stds  = stat_df["CIMO_std"].values
    rag_stds   = stat_df["RAG_std"].values

    bars1 = ax.bar(x - width/2, cimo_means, width, yerr=cimo_stds,
                   label="CIMO", color=cimo_color, alpha=0.85,
                   capsize=4, error_kw={"linewidth": 1.5})
    bars2 = ax.bar(x + width/2, rag_means, width, yerr=rag_stds,
                   label="RAG",  color=rag_color,  alpha=0.85,
                   capsize=4, error_kw={"linewidth": 1.5})

    # 标记显著性
    for i, (_, row) in enumerate(stat_df.iterrows()):
        sig = row["significance"]
        if sig != "ns":
            y_top = max(cimo_means[i] + cimo_stds[i],
                        rag_means[i]  + rag_stds[i]) + 0.1
            ax.text(i, y_top, sig, ha="center", fontsize=10,
                    color="darkgreen", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_display.values()), fontsize=9)
    ax.set_ylabel("Score (1-3 per dimension, 4-12 total)")
    ax.set_title("(a) Mean Scores by Metric")
    ax.legend()
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.grid(axis="y", alpha=0.3)

    # --- 图2: Total分数分布（箱线图）---
    ax = axes[1]
    cimo_totals = eval_df[eval_df["method"] == "CIMO"]["total"].values
    rag_totals  = eval_df[eval_df["method"] == "RAG"]["total"].values

    bp = ax.boxplot(
        [cimo_totals, rag_totals],
        labels=["CIMO", "RAG"],
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 2},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
    )
    bp["boxes"][0].set_facecolor(cimo_color)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(rag_color)
    bp["boxes"][1].set_alpha(0.7)

    # 叠加散点
    for i, (vals, color) in enumerate(
        [(cimo_totals, cimo_color), (rag_totals, rag_color)], 1
    ):
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   alpha=0.4, s=20, color=color, zorder=3)

    ax.set_ylabel("Total Score (4-12)")
    ax.set_title("(b) Total Score Distribution")
    ax.grid(axis="y", alpha=0.3)

    # --- 图3: 各维度Gap类型分布（堆叠条形）---
    ax = axes[2]
    for method_idx, (method, color) in enumerate(
        [("CIMO", cimo_color), ("RAG", rag_color)]
    ):
        mdf     = eval_df[eval_df["method"] == method]
        counts  = mdf["gap_type"].value_counts(normalize=True) * 100
        gap_types = ["persistent", "evolved", "new"]
        alphas  = [0.9, 0.65, 0.4]

        bottom = 0
        x_pos  = method_idx
        for gtype, alpha in zip(gap_types, alphas):
            val = counts.get(gtype, 0)
            ax.bar(x_pos, val, bottom=bottom,
                   color=color, alpha=alpha, width=0.5)
            if val > 5:
                ax.text(x_pos, bottom + val/2, f"{val:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
            bottom += val

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CIMO", "RAG"])
    ax.set_ylabel("Gap Type Distribution (%)")
    ax.set_title("(c) Gap Type Distribution")
    ax.set_ylim(0, 110)

    patches = [
        mpatches.Patch(color="grey", alpha=0.9, label="Persistent"),
        mpatches.Patch(color="grey", alpha=0.65, label="Evolved"),
        mpatches.Patch(color="grey", alpha=0.4,  label="New"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{EVAL_DIR}/comparison_plot.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{EVAL_DIR}/comparison_plot.pdf", bbox_inches="tight")
    print(f"✓ 图表已保存至 {EVAL_DIR}/")
    plt.show()


# ============================================================
# Step 6: 生成论文用汇总表
# ============================================================
def generate_paper_table(eval_df, stat_df):
    """生成可直接放入论文的格式化对比表"""

    rows = []
    metric_names = {
        "specificity":      "Specificity",
        "evidence_quality": "Evidence Quality",
        "actionability":    "Actionability",
        "cimo_alignment":   "CIMO Alignment",
        "total":            "Total Score",
    }

    for _, row in stat_df.iterrows():
        metric = row["Metric"]
        rows.append({
            "Criterion":              metric_names.get(metric, metric),
            "CIMO (Mean ± SD)":       f"{row['CIMO_mean']:.2f} ± {row['CIMO_std']:.2f}",
            "RAG (Mean ± SD)":        f"{row['RAG_mean']:.2f} ± {row['RAG_std']:.2f}",
            "Δ (CIMO−RAG)":           f"{row['CIMO_mean']-row['RAG_mean']:+.2f}",
            "Mann-Whitney p":         f"{row['p_value']:.4f}",
            "Significance":           row["significance"],
            "Cohen's d":              f"{row['Cohens_d']:.3f}",
        })

    table_df = pd.DataFrame(rows)
    table_df.to_excel(f"{EVAL_DIR}/paper_table_comparison.xlsx", index=False)

    print("\n=== 论文Table（直接可用）===")
    print(table_df.to_string(index=False))
    return table_df


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":

    # 1. 加载两组结果
    df = load_gaps(
        cimo_path = f"{OUTPUT_DIR}/gaps_raw_2020_2022.json",
        rag_path  = f"{RAG_DIR}/gaps_raw_rag_fulltext.json",
    )

    # 2. LLM自动评分（约5-10分钟）
    eval_df = run_judge_evaluation(df)

    # 3. 统计检验
    stat_df = run_statistical_tests(eval_df)

    # 4. 导出人工验证样本
    export_human_validation_sample(eval_df, n=30)

    # 5. 可视化
    plot_comparison(eval_df, stat_df)

    # 6. 生成论文Table
    generate_paper_table(eval_df, stat_df)

    print(f"\n✓ 全部完成，结果在 {EVAL_DIR}/")
    print("  下一步：打开 human_validation_sample.xlsx，")
    print("  在 human_* 列填入你的评分，然后运行:")
    print("  compute_kappa(f'{EVAL_DIR}/human_validation_sample.xlsx')")