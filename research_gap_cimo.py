import json
import os
import time
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
CIMO_DIR   = "/data_share_from_3090/wy_code/code/EE/NER/utd24/cimo_results_2020_2022"
OUTPUT_DIR = "/data_share_from_3090/wy_code/code/EE/NER/utd24/gap_results_cimo"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

JOURNAL_TO_DISCIPLINE = {
    "Academy_of_Management_Journal":      "Strategic_Management",
    "Strategic_Management_Journal":       "Strategic_Management",
    "Journal_of_Marketing":               "Marketing",
    "Industrial_Marketing_Management":    "Marketing",
    "Journal_of_Operations_Management":   "OM_SCM",
    "Journal_of_Supply_Chain_Management": "OM_SCM",
}

client = OpenAI(
    api_key  = "sk-e83c98ffccb645b8b51caec5ca63b717",
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = "deepseek-v3"

# ============================================================
# Step 1: 加载数据
# ============================================================
def load_all_cimo(cimo_dir):
    all_records = []
    skipped = 0

    for filepath in sorted(Path(cimo_dir).glob("results_deepseek_*.jsonl")):
        stem       = filepath.stem.replace("results_deepseek_", "")
        parts      = stem.rsplit("_", 1)
        journal    = parts[0]
        year       = int(parts[1])
        discipline = JOURNAL_TO_DISCIPLINE.get(journal, "Unknown")

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    paper = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                cio_list = paper.get("cio_list", [])
                if not cio_list:
                    skipped += 1
                    continue

                for cio in cio_list:
                    all_records.append({
                        "paper_title": paper.get("title", ""),
                        "authors":     paper.get("authors", ""),
                        "journal":     journal,
                        "year":        year,
                        "discipline":  discipline,
                        "cio_id":      cio.get("id"),
                        "C":           cio.get("context", ""),
                        "I":           cio.get("intervention", ""),
                        "M":           cio.get("mechanism", ""),
                        "O":           cio.get("outcome", ""),
                    })

    print(f"✓ 共加载 {len(all_records)} 条CIO记录（跳过 {skipped} 条异常）")
    return all_records


# ============================================================
# Step 2: 数据探查
# ============================================================
def inspect(df):
    print("\n" + "="*50)
    print("数据探查报告")
    print("="*50)
    print(f"总CIO条目数  : {len(df)}")
    print(f"论文总数     : {df['paper_title'].nunique()}")

    print("\n--- 各期刊×年份条目数 ---")
    print(df.groupby(["journal", "year"]).size().to_string())

    print("\n--- 各学科条目数 ---")
    print(df.groupby("discipline").size().to_string())

    print("\n--- CIMO各维度填充率 ---")
    for dim in ["C", "I", "M", "O"]:
        filled = df[dim].apply(
            lambda x: bool(x and str(x).strip() and str(x).strip() != "None")
        ).sum()
        print(f"  {dim}: {filled}/{len(df)} ({filled/len(df):.1%})")

    print("\n--- 每篇论文平均CIO条目数 ---")
    per_paper = df.groupby("paper_title").size()
    print(f"  均值={per_paper.mean():.2f}, 最大={per_paper.max()}, 最小={per_paper.min()}")
    print("="*50 + "\n")


# ============================================================
# Step 3: Gap 抽取
# ============================================================
TANSKANEN_GAPS = """
Tanskanen et al. (2017) identified these gaps in 1997-2012 ERM research:
   1. [C] We find a need to broaden the perspective of research from dyads, chains and networks to whole industries.
   2. [C] We find a need to focus the perspective to better understand the idiosyncrasies of managing suppliers in different (purchasing) categories.
   3. [C、M] Although scholars agree that companies can benefit immensely from a well-designed configuration of an alliance network, in particular in new product development, we still know surprisingly little of  how to adapt the network to the changes in the industry.
   4. [C] Management of external resources is still rarely  studied at the category level.
   5. [C] There is a dearth of studies of external resource management in the public sector.
   6. [C、I] There is a need for more studies that consider the idiosyncrasies of the public sector in decisions of governance mode and mechanism.
   7. [C] There is a need for more studies that consider the idiosyncrasies of the public sector in network formation and relationship initiation.
   8. [C] There is a need for more studies that consider the idiosyncrasies of the public sector in interorganizational relationships.
   9. [C、M] The studies in the new product development context remain relatively silent on situations where firms have low R&D capacity and, therefore, are limited  in their abilities in learning new technological insights, and implementing them in innovation and  new business development.
   10. [C、M]:It would be interesting to investigate how absorptive capacity manifests itself in low R&D contexts.
"""

GAP_PROMPT = """
You are an expert in Evidence-Based Management (EBM) research synthesis.

## Discipline: {discipline}
## Papers: {n_papers} papers, {n_cio} CIO entries
## Period: 2020-2022

## CIMO-structured evidence (Context | Intervention | Mechanism | Outcome):
{cimo_text}

## Known gaps from 1997-2012 baseline (Tanskanen et al., 2017):
{tanskanen_gaps}

## Your Task:
Analyze the CIMO evidence above and identify research gaps.
relationships but leave the underlying generative process unexplained.

### STRICT RULES you must follow:
1. "dimension" MUST be exactly one of: "C", "I", "M", "O"
   — NEVER use "Method" or any other label
2. Every "candidate_proposition" must use concrete constructs
   found in the retrieved passages — no placeholder text
3. "evidence_pattern" must quote or closely paraphrase
   specific language from the retrieved passages
4. Classify gap_type:
   - "persistent": unaddressed since Tanskanen 2017
   - "evolved": Tanskanen identified it, now in new form
   - "new": not in Tanskanen's list


### Output format — return ONLY a valid JSON array, no markdown:
[
  {{
    "dimension": "C|I|M|O",
    "gap_type": "persistent|evolved|new",
    "gap_statement": "One precise sentence: what is missing and why it matters",
    "evidence_pattern": "Direct evidence from the retrieved passages",
    "related_tanskanen_gap": "Gap number 1-10, or null if new",
    "candidate_proposition": "Concrete proposition with real constructs from the passages"
  }}
]
"""

def format_cimo_text(group_df, max_entries=60):
    if len(group_df) > max_entries:
        group_df = group_df.sample(max_entries, random_state=42)
    lines = []
    for _, row in group_df.iterrows():
        title = str(row["paper_title"])[:50]
        c = str(row["C"] or "—")[:150]
        i = str(row["I"] or "—")[:150]
        m = str(row["M"] or "—")[:150]
        o = str(row["O"] or "—")[:150]
        lines.append(
            f"[{title}]\n"
            f"  C: {c}\n"
            f"  I: {i}\n"
            f"  M: {m}\n"
            f"  O: {o}"
        )
    return "\n\n".join(lines)


def extract_gaps(discipline, group_df):
    n_papers  = group_df["paper_title"].nunique()
    n_cio     = len(group_df)
    cimo_text = format_cimo_text(group_df)

    prompt = GAP_PROMPT.format(
        discipline     = discipline,
        n_papers       = n_papers,
        n_cio          = n_cio,
        cimo_text      = cimo_text,
        tanskanen_gaps = TANSKANEN_GAPS
    )

    try:
        response = client.chat.completions.create(
            model       = MODEL,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 3000,
            temperature = 0.1,
        )
        raw = response.choices[0].message.content.strip()

        # 清理可能的markdown包裹
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.split("```")[0].strip()

        gaps = json.loads(raw)
        print(f"  ✓ [{discipline}] → {len(gaps)} 个gap")
        return gaps

    except json.JSONDecodeError as e:
        print(f"  ✗ [{discipline}] JSON解析失败: {e}")
        print(f"  原始输出前300字: {raw[:300]}")
        return []
    except Exception as e:
        print(f"  ✗ [{discipline}] API调用失败: {e}")
        return []


# ============================================================
# Step 4: 整理输出Table
# ============================================================
def build_gap_table(all_gaps, output_dir):
    rows = []
    for discipline, gaps in all_gaps.items():
        for g in gaps:
            rows.append({
                "Discipline":            discipline,
                "Dimension":             g.get("dimension", ""),
                "Type":                  g.get("gap_type", ""),
                "Gap Statement":         g.get("gap_statement", ""),
                "Evidence Pattern":      g.get("evidence_pattern", ""),
                "Tanskanen Gap #":       g.get("related_tanskanen_gap", ""),
                "Candidate Proposition": g.get("candidate_proposition", ""),
            })

    gap_df = pd.DataFrame(rows)

    pivot = pd.crosstab(
        gap_df["Dimension"],
        gap_df["Type"],
        margins=True,
        margins_name="Total"
    )
    print("\n=== Gap分布矩阵 ===")
    print(pivot)

    gap_df.to_excel(f"{output_dir}/gap_taxonomy_2020_2022.xlsx", index=False)
    pivot.to_excel(f"{output_dir}/gap_distribution_matrix.xlsx")
    print(f"\n✓ 结果已保存至 {output_dir}/")

    return gap_df, pivot


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":

    # 1. 加载
    records = load_all_cimo(CIMO_DIR)
    df = pd.DataFrame(records)

    # 2. 探查
    inspect(df)

    # 3. Gap抽取
    all_gaps = {}
    for discipline in ["Strategic_Management", "Marketing", "OM_SCM"]:
        group_df = df[df["discipline"] == discipline]
        print(f"\n处理: {discipline} "
              f"({group_df['paper_title'].nunique()}篇 / {len(group_df)}条CIO)")
        all_gaps[discipline] = extract_gaps(discipline, group_df)
        time.sleep(2)

    # 保存原始JSON
    with open(f"{OUTPUT_DIR}/gaps_raw_2020_2022.json", "w", encoding="utf-8") as f:
        json.dump(all_gaps, f, indent=2, ensure_ascii=False)

    # 4. 整理Table
    gap_df, pivot = build_gap_table(all_gaps, OUTPUT_DIR)