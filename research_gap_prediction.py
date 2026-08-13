import json
import os
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ============================================================
# 配置
# ============================================================
# 2023-2025全文目录（与2020-2022同结构）
FULLTEXT_DIR_NEW = "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/output/EBM"
TARGET_YEARS_NEW = {2023, 2024, 2025}

# 已有Gap结果

CIMO_GAP_PATH = "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/gap_results_cimo/gaps_raw_2020_2022.json"
RAG_GAP_PATH  = "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/gap_results_rag/gaps_raw_rag_fulltext.json"

OUTPUT_DIR = "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/gap_validation"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

JOURNAL_TO_DISCIPLINE = {
    "Academy_of_Management_Journal":      "Strategic_Management",
    "Strategic_Management_Journal":       "Strategic_Management",
    "Journal_of_Marketing":               "Marketing",
    "Industrial_Marketing_Management":    "Marketing",
    "Journal_of_Operations_Management":   "OM_SCM",
    "Journal_of_Supply_Chain_Management": "OM_SCM",
}

# client = OpenAI(
#     api_key  = os.environ.get("DASHSCOPE_API_KEY"),
#     base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
# )
# JUDGE_MODEL = "deepseek-v3"
client = OpenAI(
    api_key  = "None",
    base_url = "http://172.17.65.42:8001/v1",
)
JUDGE_MODEL = "Qwen3-Next-80B-A3B-Instruct"

LOCAL_MODEL_CANDIDATES = [
    "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/BAAI/bge-large-en-v1___5",

]

# ============================================================
# Step 1: 加载Embedding模型
# ============================================================
def load_embed_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for path in LOCAL_MODEL_CANDIDATES:
        try:
            model = SentenceTransformer(path, device=device)
            print(f"✓ 加载模型: {path} (device={device})")
            return model
        except Exception:
            continue
    raise RuntimeError("所有候选模型加载失败")


def encode(texts, model, batch_size=64):
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()),
                        dtype=np.float32)
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ============================================================
# Step 2: 加载2023-2025全文论文
# ============================================================
def clean_md(text):
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_papers(fulltext_dir, target_years):
    papers = []
    root   = Path(fulltext_dir)

    for journal_dir in sorted(root.iterdir()):
        if not journal_dir.is_dir():
            continue
        journal    = journal_dir.name
        discipline = JOURNAL_TO_DISCIPLINE.get(journal, "Unknown")
        if discipline == "Unknown":
            continue

        for year_dir in sorted(journal_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            try:
                year = int(year_dir.name)
            except ValueError:
                continue
            if year not in target_years:
                continue

            for paper_dir in sorted(year_dir.iterdir()):
                if not paper_dir.is_dir():
                    continue
                hybrid_dir = paper_dir / "hybrid_auto"
                if not hybrid_dir.exists():
                    continue
                md_files = list(hybrid_dir.glob("*.md"))
                if not md_files:
                    continue
                try:
                    text = md_files[0].read_text(encoding="utf-8").strip()
                    text = clean_md(text)
                except Exception:
                    continue
                if len(text) < 100:
                    continue

                papers.append({
                    "paper_id":   paper_dir.name,
                    "journal":    journal,
                    "year":       year,
                    "discipline": discipline,
                    "text":       text[:8000],  # 截断避免超长
                })

    print(f"✓ 加载2023-2025论文: {len(papers)}篇")
    by_disc = {}
    for p in papers:
        by_disc[p["discipline"]] = by_disc.get(p["discipline"], 0) + 1
    for d, c in sorted(by_disc.items()):
        print(f"   {d}: {c}篇")
    return papers


# ============================================================
# Step 3: 构建2023-2025向量索引
# ============================================================
def build_paper_index(papers, embed_model):
    """
    每篇论文取前512词作为代表向量
    粒度用论文级别（不分块），因为目标是判断"哪篇论文填补了这个gap"
    """
    print(f"\n为 {len(papers)} 篇论文生成Embedding...")
    texts = []
    for p in papers:
        words = p["text"].split()[:512]
        texts.append(" ".join(words))

    vectors = encode(texts, embed_model, batch_size=32)
    print(f"✓ 论文向量矩阵: {vectors.shape}")
    return vectors


# ============================================================
# Step 4: 候选匹配（向量召回）
# ============================================================
def retrieve_candidate_papers(gap_text, paper_vectors, papers,
                               discipline, embed_model, top_k=5):
    """
    对单个gap，在同学科2023-2025论文中找最相似的top_k篇
    """
    disc_idx     = [i for i, p in enumerate(papers)
                    if p["discipline"] == discipline]
    if not disc_idx:
        return []

    disc_vectors = paper_vectors[disc_idx]
    disc_papers  = [papers[i] for i in disc_idx]

    q_vec  = encode([gap_text], embed_model)
    scores = (disc_vectors @ q_vec.T).flatten()
    top_i  = np.argsort(scores)[::-1][:top_k]

    return [(disc_papers[i], float(scores[i])) for i in top_i]


# ============================================================
# Step 5: LLM验证是否"填补"了gap（核心判断）
# ============================================================
VERIFY_PROMPT = """
You are an expert in Evidence-Based Management (EBM) research synthesis.

## Research Gap (identified from 2020-2022 literature):
Dimension: {dimension}
Gap Statement: {gap_statement}
Candidate Proposition: {candidate_proposition}

## Paper to evaluate (published {year}):
{paper_excerpt}

## Task:
Does this paper substantively ADDRESS the research gap above?

Scoring rubric:
- 3 = Directly addresses: paper provides empirical or theoretical evidence
      that fills this specific gap
- 2 = Partially addresses: paper touches on the gap topic but does not
      fully resolve it
- 1 = Does not address: paper is related in topic but does not contribute
      to closing this gap
- 0 = Irrelevant: paper is unrelated to the gap

Return ONLY valid JSON:
{{
  "score": <0|1|2|3>,
  "addressed": <true if score >= 2, else false>,
  "rationale": "<one sentence explaining your judgment>"
}}
"""

def verify_gap_coverage(gap, candidate_papers, threshold_score=2):
    """
    对一个gap，逐一验证候选论文是否填补了它
    返回：最高分、是否被填补、填补论文列表
    """
    gap_text = gap.get("gap_statement", "")
    prop     = gap.get("candidate_proposition", "")
    dim      = gap.get("dimension", "")

    best_score    = 0
    addressed_by  = []

    for paper, sim_score in candidate_papers:
        excerpt = " ".join(paper["text"].split()[:400])

        prompt = VERIFY_PROMPT.format(
            dimension             = dim,
            gap_statement         = gap_text[:250],
            candidate_proposition = prop[:200],
            year                  = paper["year"],
            paper_excerpt         = excerpt,
        )

        for attempt in range(2):
            try:
                resp = client.chat.completions.create(
                    model       = JUDGE_MODEL,
                    messages    = [{"role": "user", "content": prompt}],
                    max_tokens  = 200,
                    temperature = 0.0,
                )
                raw = resp.choices[0].message.content.strip()
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.split("```")[0].strip()

                result = json.loads(raw)
                score  = int(result.get("score", 0))

                if score > best_score:
                    best_score = score
                if score >= threshold_score:
                    addressed_by.append({
                        "paper_id":  paper["paper_id"],
                        "journal":   paper["journal"],
                        "year":      paper["year"],
                        "sim_score": round(sim_score, 3),
                        "llm_score": score,
                        "rationale": result.get("rationale", ""),
                    })
                break

            except Exception:
                if attempt == 0:
                    time.sleep(2)

        time.sleep(0.5)

    is_addressed = best_score >= threshold_score
    return best_score, is_addressed, addressed_by


# ============================================================
# Step 6: 主验证循环
# ============================================================
def run_validation(gap_file, method_label,
                   papers_2325, paper_vectors, embed_model,
                   top_k=5):
    with open(gap_file, encoding="utf-8") as f:
        all_gaps = json.load(f)

    results = []
    total   = sum(len(v) for v in all_gaps.values())
    done    = 0

    for discipline, gaps in all_gaps.items():
        print(f"\n[{method_label}] 处理: {discipline} ({len(gaps)}个gap)")

        for gap in gaps:
            dim        = gap.get("dimension", "")
            gap_stmt   = gap.get("gap_statement", "")
            gap_type   = gap.get("gap_type", "")

            if not gap_stmt or dim not in ["C", "I", "M", "O"]:
                done += 1
                continue

            # 向量召回候选论文
            query = f"{gap_stmt} {gap.get('candidate_proposition', '')}"
            candidates = retrieve_candidate_papers(
                query, paper_vectors, papers_2325,
                discipline, embed_model, top_k=top_k
            )

            # LLM验证
            best_score, is_addressed, addressed_by = verify_gap_coverage(
                gap, candidates
            )

            results.append({
                "method":            method_label,
                "discipline":        discipline,
                "dimension":         dim,
                "gap_type":          gap_type,
                "gap_statement":     gap_stmt,
                "best_llm_score":    best_score,
                "is_addressed":      is_addressed,
                "n_addressing_papers": len(addressed_by),
                "addressing_papers": json.dumps(addressed_by, ensure_ascii=False),
            })

            done += 1
            status = "✓ 已填补" if is_addressed else "○ 未填补"
            print(f"  [{done}/{total}] [{dim}][{gap_type}] {status} "
                  f"(score={best_score}) {gap_stmt[:60]}")

    return pd.DataFrame(results)


# ============================================================
# Step 7: 统计对比与可视化
# ============================================================
def analyze_coverage(cimo_df, rag_df):
    combined = pd.concat([cimo_df, rag_df], ignore_index=True)
    combined.to_excel(f"{OUTPUT_DIR}/validation_details.xlsx", index=False)

    print("\n" + "="*55)
    print("预测覆盖率对比")
    print("="*55)

    # --- 总体覆盖率 ---
    summary_rows = []
    
    for method, df in [("CIMO", cimo_df), ("RAG", rag_df)]:
        total      = len(df)
        addressed  = df["is_addressed"].sum()
        rate       = addressed / total * 100 if total > 0 else 0
        print(f"\n[{method}]")
        print(f"  总Gap数:     {total}")
        print(f"  被填补数:    {addressed}")
        print(f"  覆盖率:      {rate:.1f}%")
        summary_dims = []
        for dim in ["C", "I", "M", "O"]:
            sub   = df[df["dimension"] == dim]
            if len(sub) == 0:
                continue
            d_rate = sub["is_addressed"].mean() * 100
            print(f"  [{dim}] {len(sub)}个gap, 覆盖率={d_rate:.1f}%")
            summary_dims.append({
                "Method":         method,
                "Dimension":      dim,
                "Total_Gaps":     len(sub),
                "Addressed":      sub["is_addressed"].sum(),
                "Coverage_Rate":  round(d_rate, 1),
            })

        summary_rows.append({
            "Method":         method,
            "Total_Gaps":     total,
            "Addressed":      addressed,
            "Coverage_Rate":  round(rate, 1),

        })
        

    # 卡方检验：两组覆盖率是否显著不同
    cimo_addr = cimo_df["is_addressed"].sum()
    cimo_not  = len(cimo_df) - cimo_addr
    rag_addr  = rag_df["is_addressed"].sum()
    rag_not   = len(rag_df) - rag_addr

    contingency = np.array([[cimo_addr, cimo_not],
                             [rag_addr,  rag_not]])
    chi2, p_chi2, _, _ = stats.chi2_contingency(contingency)
    print(f"\n卡方检验: χ²={chi2:.3f}, p={p_chi2:.4f}")
    sig = "***" if p_chi2 < 0.001 else ("**" if p_chi2 < 0.01
          else ("*" if p_chi2 < 0.05 else "ns"))
    print(f"显著性: {sig}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_excel(f"{OUTPUT_DIR}/coverage_summary.xlsx", index=False)
    summary_dims_df = pd.DataFrame(summary_dims)
    summary_dims_df.to_excel(f"{OUTPUT_DIR}/coverage_summary_dims.xlsx", index=False)



    # --- 可视化 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Gap Prediction Validation: CIMO vs RAG (2023-2025 Papers)",
                 fontsize=13, fontweight="bold")

    cimo_color = "#2196F3"
    rag_color  = "#FF5722"

    # 图1：总体覆盖率
    ax = axes[0]
    methods = ["CIMO", "RAG"]
    rates   = [cimo_df["is_addressed"].mean()*100,
               rag_df["is_addressed"].mean()*100]
    bars = ax.bar(methods, rates,
                  color=[cimo_color, rag_color], alpha=0.85, width=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", fontweight="bold")
    ax.text(0.5, max(rates) + 8,
            f"χ²={chi2:.2f}, p={p_chi2:.4f} {sig}",
            ha="center", transform=ax.transAxes, fontsize=9,
            color="darkgreen")
    ax.set_ylabel("Coverage Rate (%)")
    ax.set_title("(a) Overall Gap Coverage Rate")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    # 图2：各维度覆盖率
    ax = axes[1]
    dims      = ["C", "I", "M", "O"]
    x         = np.arange(len(dims))
    width     = 0.35
    cimo_rates = [cimo_df[cimo_df["dimension"]==d]["is_addressed"].mean()*100
                  if len(cimo_df[cimo_df["dimension"]==d]) > 0 else 0
                  for d in dims]
    rag_rates  = [rag_df[rag_df["dimension"]==d]["is_addressed"].mean()*100
                  if len(rag_df[rag_df["dimension"]==d]) > 0 else 0
                  for d in dims]

    ax.bar(x - width/2, cimo_rates, width, label="CIMO",
           color=cimo_color, alpha=0.85)
    ax.bar(x + width/2, rag_rates,  width, label="RAG",
           color=rag_color,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Coverage Rate (%)")
    ax.set_title("(b) Coverage Rate by CIMO Dimension")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 图3：Gap类型覆盖率
    ax = axes[2]
    gap_types  = ["persistent", "evolved", "new"]
    x          = np.arange(len(gap_types))
    cimo_grates = [
        cimo_df[cimo_df["gap_type"]==t]["is_addressed"].mean()*100
        if len(cimo_df[cimo_df["gap_type"]==t]) > 0 else 0
        for t in gap_types
    ]
    rag_grates  = [
        rag_df[rag_df["gap_type"]==t]["is_addressed"].mean()*100
        if len(rag_df[rag_df["gap_type"]==t]) > 0 else 0
        for t in gap_types
    ]

    ax.bar(x - width/2, cimo_grates, width, label="CIMO",
           color=cimo_color, alpha=0.85)
    ax.bar(x + width/2, rag_grates,  width, label="RAG",
           color=rag_color,  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["Persistent", "Evolved", "New"])
    ax.set_ylabel("Coverage Rate (%)")
    ax.set_title("(c) Coverage Rate by Gap Type")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/validation_plot.png", dpi=300,
                bbox_inches="tight")
    plt.savefig(f"{OUTPUT_DIR}/validation_plot.pdf", bbox_inches="tight")
    print(f"\n✓ 图表已保存")
    plt.show()

    return summary_df


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":

    # 1. 加载embedding模型
    embed_model = load_embed_model()

    # 2. 加载2023-2025全文
    papers_2325 = load_papers(FULLTEXT_DIR_NEW, TARGET_YEARS_NEW)
    if not papers_2325:
        raise RuntimeError("未找到2023-2025论文，检查路径")

    # 3. 构建2023-2025向量索引（缓存）
    VEC_CACHE   = f"{OUTPUT_DIR}/paper_vectors_2325.npy"
    META_CACHE  = f"{OUTPUT_DIR}/papers_meta_2325.json"

    if Path(VEC_CACHE).exists() and Path(META_CACHE).exists():
        print("✓ 加载缓存向量")
        paper_vectors = np.load(VEC_CACHE)
        with open(META_CACHE) as f:
            papers_2325 = json.load(f)
    else:
        paper_vectors = build_paper_index(papers_2325, embed_model)
        np.save(VEC_CACHE, paper_vectors)
        with open(META_CACHE, "w") as f:
            json.dump(papers_2325, f, ensure_ascii=False)
        print("✓ 向量已缓存")

    # 4. CIMO验证
    print("\n===== 开始CIMO Gap验证 =====")
    cimo_df = run_validation(
        CIMO_GAP_PATH, "CIMO",
        papers_2325, paper_vectors, embed_model, top_k=5
    )
    cimo_df.to_excel(f"{OUTPUT_DIR}/cimo_validation.xlsx", index=False)

    # 5. RAG验证
    print("\n===== 开始RAG Gap验证 =====")
    rag_df = run_validation(
        RAG_GAP_PATH, "RAG",
        papers_2325, paper_vectors, embed_model, top_k=5
    )
    rag_df.to_excel(f"{OUTPUT_DIR}/rag_validation.xlsx", index=False)

    # 6. 统计对比
    summary_df = analyze_coverage(cimo_df, rag_df)

    print("\n✓ 全部完成，结果在:", OUTPUT_DIR)