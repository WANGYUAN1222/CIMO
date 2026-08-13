import json
import os
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ============================================================
# 配置
# ============================================================
# 全文根目录（包含EBM子目录）
FULLTEXT_DIR = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/EBM"
OUTPUT_DIR   = "/data_share_from_3090/wy_code/code/EE/NER/utd24/gap_results_rag"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

TARGET_YEARS = {2020, 2021, 2022}   # 只处理2020-2022

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
CHAT_MODEL      = "deepseek-v3"
EMBEDDING_MODEL = "text-embedding-v3"


# ============================================================
# Step 1: 加载全文 .md 文件
# ============================================================
def load_fulltext_papers(fulltext_dir, target_years):
    """
    遍历 output/EBM/{Journal}/{Year}/{paper_id}/hybrid_auto/{paper_id}.md
    只加载 target_years 中的年份
    """
    papers  = []
    skipped = 0

    ebm_root = Path(fulltext_dir)

    for journal_dir in sorted(ebm_root.iterdir()):
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

            # 遍历每篇论文目录
            for paper_dir in sorted(year_dir.iterdir()):
                if not paper_dir.is_dir():
                    continue
                paper_id   = paper_dir.name
                hybrid_dir = paper_dir / "hybrid_auto"
                if not hybrid_dir.exists():
                    skipped += 1
                    continue

                # 找到 .md 文件
                md_files = list(hybrid_dir.glob("*.md"))
                if not md_files:
                    skipped += 1
                    continue

                md_path = md_files[0]   # 每个目录只有一个.md
                try:
                    text = md_path.read_text(encoding="utf-8").strip()
                except Exception as e:
                    print(f"  ⚠ 读取失败: {md_path} → {e}")
                    skipped += 1
                    continue

                if len(text) < 100:     # 内容太短跳过
                    skipped += 1
                    continue

                papers.append({
                    "paper_id":   paper_id,
                    "journal":    journal,
                    "year":       year,
                    "discipline": discipline,
                    "md_path":    str(md_path),
                    "text":       text,
                })

    print(f"✓ 共加载 {len(papers)} 篇全文论文（跳过 {skipped} 个目录）")
    # 按学科统计
    disc_count = {}
    for p in papers:
        disc_count[p["discipline"]] = disc_count.get(p["discipline"], 0) + 1
    for d, c in disc_count.items():
        print(f"   {d}: {c}篇")
    return papers


# ============================================================
# Step 2: 文本分块
# ============================================================
def clean_markdown(text: str) -> str:
    """清理markdown标记，保留纯文本"""
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)   # 图片
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)     # 链接
    text = re.sub(r"#{1,6}\s*", "", text)          # 标题符号
    text = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", text)  # 粗/斜体
    text = re.sub(r"\n{3,}", "\n\n", text)         # 多余空行
    return text.strip()


def chunk_papers(papers, chunk_size=400, overlap=80):
    """
    按词切块，保留来源信息
    全文比摘要长很多，chunk_size适当加大
    """
    chunks  = []
    too_short = 0

    for paper in papers:
        text  = clean_markdown(paper["text"])
        words = text.split()
        step  = max(chunk_size - overlap, 1)

        for start in range(0, max(len(words), 1), step):
            chunk_words = words[start: start + chunk_size]
            if len(chunk_words) < 30:   # 太短丢弃
                too_short += 1
                continue
            chunks.append({
                "chunk_id":   f"{paper['paper_id']}_{start}",
                "paper_id":   paper["paper_id"],
                "journal":    paper["journal"],
                "year":       paper["year"],
                "discipline": paper["discipline"],
                "text":       " ".join(chunk_words),
            })

    print(f"✓ 共生成 {len(chunks)} 个文本块（丢弃过短块 {too_short} 个）")
    per_paper = len(chunks) / max(len(papers), 1)
    print(f"   每篇平均 {per_paper:.1f} 个块")
    return chunks


# ============================================================
# Step 3: Embedding + 向量索引
# ============================================================
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

# ============================================================
# 本地Embedding配置（替换原来的EMBEDDING_MODEL和get_embeddings_batch）
# ============================================================

# 按优先级选择：服务器上有哪个用哪个
LOCAL_MODEL_CANDIDATES = [
    # 你的BAAI目录（从截图看到了BAAI文件夹）
    "/data_share_from_3090/wy_code/code/EE/NER/utd24/BAAI/bge-large-en-v1.5",
]

def load_local_embed_model():
    """按优先级加载第一个可用的本地模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    for model_path in LOCAL_MODEL_CANDIDATES:
        try:
            model = SentenceTransformer(model_path, device=device)
            dim   = model.get_sentence_embedding_dimension()
            print(f"✓ 加载模型: {model_path}  (维度={dim}, 设备={device})")
            return model
        except Exception as e:
            print(f"  跳过 {model_path}: {e}")

    raise RuntimeError("所有候选模型均加载失败，请检查路径")


def get_embeddings_batch(texts, model, batch_size=64):
    """
    本地模型批量encode，替换原来的API调用版本
    不需要sleep，速度远快于API
    """
    if not texts:
        return np.zeros((0, model.get_sentence_embedding_dimension()),
                        dtype=np.float32)

    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch   = texts[i: i + batch_size]
        vectors = model.encode(
            batch,
            normalize_embeddings = True,   # 直接归一化，内积=余弦相似度
            show_progress_bar    = False,
            convert_to_numpy     = True,
        )
        all_vectors.append(vectors)
        if (i // batch_size) % 10 == 0:
            print(f"  Embedding进度: {min(i+batch_size, len(texts))}/{len(texts)}")

    return np.vstack(all_vectors).astype(np.float32)


def build_index(chunks, embed_model):
    """构建向量索引（已归一化，直接内积=余弦相似度）"""
    print(f"\n正在生成 {len(chunks)} 个块的Embedding...")
    texts   = [c["text"] for c in chunks]
    vectors = get_embeddings_batch(texts, embed_model, batch_size=64)
    print(f"✓ Embedding完成，矩阵形状: {vectors.shape}")
    return vectors


def retrieve_for_discipline(query, vectors, chunks, discipline,
                             embed_model, top_k=10):
    """检索指定学科内最相关的chunks"""
    disc_idx     = [i for i, c in enumerate(chunks)
                    if c["discipline"] == discipline]
    disc_vectors = vectors[disc_idx]
    disc_chunks  = [chunks[i] for i in disc_idx]

    q_vec   = get_embeddings_batch([query], embed_model)  # 已归一化
    scores  = (disc_vectors @ q_vec.T).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [(disc_chunks[i], float(scores[i])) for i in top_idx]


# ============================================================
# Step 4: RAG Gap 抽取
# ============================================================
TANSKANEN_GAPS = """
Tanskanen et al. (2017) identified these gaps in 1997-2012 ERM research:
   1. [C] We find a need to broaden the perspective of research from dyads, chains and networks to whole industries.
   2. [C] We find a need to focus the perspective to better understand the idiosyncrasies of managing suppliers in different (purchasing) categories.
   3. [C、M] Although scholars agree that companies can benefit immensely from a well-designed configuration of an alliance network, in particular in new product development, we still know surprisingly little of how to adapt the network to the changes in the industry.
   4. [C] Management of external resources is still rarely studied at the category level.
   5. [C] There is a dearth of studies of external resource management in the public sector.
   6. [C、I] There is a need for more studies that consider the idiosyncrasies of the public sector in decisions of governance mode and mechanism.
   7. [C] There is a need for more studies that consider the idiosyncrasies of the public sector in network formation and relationship initiation.
   8. [C] There is a need for more studies that consider the idiosyncrasies of the public sector in interorganizational relationships.
   9. [C、M] The studies in the new product development context remain relatively silent on situations where firms have low R&D capacity.
   10. [C、M] It would be interesting to investigate how absorptive capacity manifests itself in low R&D contexts.
"""

DISCIPLINE_QUERIES = {
    "Strategic_Management": [
        "alliance governance mechanism external resources strategy",
        "network formation interorganizational relationship firm performance",
        "knowledge transfer learning absorptive capacity",
        "open innovation external partner R&D",
        "public sector governance strategic alliance",
        "digital platform ecosystem governance",
        "geopolitical risk supply chain strategy",
    ],
    "Marketing": [
        "buyer supplier relationship trust commitment marketing",
        "customer involvement new product development",
        "supply chain collaboration value creation",
        "digital B2B platform relationship marketing",
        "sustainability ESG supply chain marketing",
        "servitization service infusion external partner",
        "cross-border e-commerce platform selection",
    ],
    "OM_SCM": [
        "supply chain integration operational performance supplier",
        "supplier development information sharing coordination",
        "digital technology supply chain management automation",
        "supply chain risk disruption mitigation",
        "sustainable supply chain ESG practices",
        "platform governance supplier participation",
        "reshoring nearshoring supply chain restructuring",
    ],
}

RAG_GAP_PROMPT = """
You are an expert in Evidence-Based Management (EBM) research synthesis.

## Discipline: {discipline}
## Period: 2020-2022 (full-text papers)
## Retrieved passages ({n_chunks} chunks from {n_papers} papers):

{retrieved_text}

## Known gaps from 1997-2012 baseline (Tanskanen et al., 2017):
{tanskanen_gaps}

## Your Task:
Based ONLY on the retrieved full-text passages above, identify research
gaps in External Resource Management (ERM) for 2020-2022.

A research gap exists when:
- A topic is mentioned but its mechanism is left unexplained (M gap)
- A context/population is acknowledged but not studied (C gap)
- An intervention is proposed but not empirically tested (I gap)
- An outcome is assumed but not measured (O gap)

### STRICT RULES:
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


Return ONLY a valid JSON array, no markdown:
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

PLACEHOLDER_RE = re.compile(
    r"context x|intervention y|mechanism z|outcome o", re.IGNORECASE
)

def parse_json_safe(raw: str) -> list:
    if "```" in raw:
        for block in raw.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("["):
                raw = block
                break
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start: end + 1]
    return json.loads(raw)


def validate_gaps(gaps, discipline):
    valid_dims  = {"C", "I", "M", "O"}
    valid_types = {"persistent", "evolved", "new"}
    cleaned = []
    for g in gaps:
        dim      = str(g.get("dimension", "")).strip().upper()
        gap_type = str(g.get("gap_type", "")).strip().lower()
        prop     = str(g.get("candidate_proposition", ""))

        if dim not in valid_dims:
            print(f"  ⚠ [{discipline}] 非法dimension '{dim}' → 跳过")
            continue
        if gap_type not in valid_types:
            gap_type = "new"
        if PLACEHOLDER_RE.search(prop):
            g["candidate_proposition"] = "[TODO: 需人工补充]"
            print(f"  ⚠ [{discipline}] 发现占位符 → 标记TODO")

        g["dimension"] = dim
        g["gap_type"]  = gap_type
        cleaned.append(g)
    return cleaned


def format_retrieved(retrieved, max_chars=5000):
    lines = []
    total = 0
    seen  = set()
    for chunk, score in retrieved:
        pid = chunk["paper_id"]
        if pid in seen:
            continue
        seen.add(pid)
        snippet = chunk["text"][:400]
        line    = (f"[{chunk['journal']} {chunk['year']} | score={score:.3f}]\n"
                   f"{snippet}")
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    return "\n\n".join(lines)


def extract_gaps_rag(discipline, vectors, chunks, embed_model, max_retries=2):
    queries = DISCIPLINE_QUERIES[discipline]

    # 多query检索，合并去重
    seen_chunks = {}
    for query in queries:
        results = retrieve_for_discipline(query, vectors, chunks, discipline, embed_model, top_k=8)
        for chunk, score in results:
            cid = chunk["chunk_id"]
            if cid not in seen_chunks or score > seen_chunks[cid][1]:
                seen_chunks[cid] = (chunk, score)
        time.sleep(0.2)

    # 按相似度排序，取top 25
    sorted_results = sorted(seen_chunks.values(), key=lambda x: x[1], reverse=True)[:25]
    n_papers       = len({c["paper_id"] for c, _ in sorted_results})
    retrieved_text = format_retrieved(sorted_results)

    prompt = RAG_GAP_PROMPT.format(
        discipline     = discipline,
        n_chunks       = len(sorted_results),
        n_papers       = n_papers,
        retrieved_text = retrieved_text,
        tanskanen_gaps = TANSKANEN_GAPS,
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model       = CHAT_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                max_tokens  = 3000,
                temperature = 0.1,
            )
            raw  = response.choices[0].message.content.strip()
            gaps = parse_json_safe(raw)
            gaps = validate_gaps(gaps, discipline)
            print(f"  ✓ [{discipline}] 第{attempt}次 → {len(gaps)} 个gap")
            return gaps

        except json.JSONDecodeError as e:
            print(f"  ✗ [{discipline}] 第{attempt}次JSON失败: {e}")
            if attempt < max_retries:
                time.sleep(3)
        except Exception as e:
            print(f"  ✗ [{discipline}] 第{attempt}次API失败: {e}")
            if attempt < max_retries:
                time.sleep(5)

    return []


# ============================================================
# Step 5: 整理输出
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
                "Method":                "RAG_Fulltext",
            })

    gap_df = pd.DataFrame(rows)

    illegal = gap_df[~gap_df["Dimension"].isin(["C", "I", "M", "O"])]
    if not illegal.empty:
        print(f"\n⚠ 仍有 {len(illegal)} 条非法dimension，需人工检查")

    pivot = pd.crosstab(
        gap_df["Dimension"],
        gap_df["Type"],
        margins=True,
        margins_name="Total"
    )
    print("\n=== RAG(全文) Gap分布矩阵 ===")
    print(pivot)

    gap_df.to_excel(f"{output_dir}/gap_taxonomy_rag_fulltext.xlsx",    index=False)
    pivot.to_excel(f"{output_dir}/gap_distribution_matrix_rag_fulltext.xlsx")
    print(f"\n✓ 结果已保存至 {output_dir}/")
    return gap_df, pivot


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    # 1. 加载全文
    papers = load_fulltext_papers(FULLTEXT_DIR, TARGET_YEARS)

    # 2. 分块
    chunks = chunk_papers(papers, chunk_size=400, overlap=80)

    # ★ 改动1：加载本地模型（原来是配置EMBEDDING_MODEL字符串）
    embed_model = load_local_embed_model()

    # 3. 构建向量索引
    CACHE_VEC   = f"{OUTPUT_DIR}/chunk_vectors.npy"
    CACHE_CHUNK = f"{OUTPUT_DIR}/chunks_meta.json"

    if Path(CACHE_VEC).exists() and Path(CACHE_CHUNK).exists():
        print("✓ 发现缓存，直接加载")
        vectors = np.load(CACHE_VEC)
        with open(CACHE_CHUNK) as f:
            chunks = json.load(f)
    else:
        # ★ 改动2：build_index 传入 embed_model
        vectors = build_index(chunks, embed_model)
        np.save(CACHE_VEC, vectors)
        with open(CACHE_CHUNK, "w") as f:
            json.dump(chunks, f, ensure_ascii=False)
        print("✓ 向量索引已缓存")

    # 4. Gap抽取
    all_gaps = {}
    for discipline in ["Strategic_Management", "Marketing", "OM_SCM"]:
        n = sum(1 for p in papers if p["discipline"] == discipline)
        print(f"\n处理: {discipline} ({n}篇)")
        # ★ 改动3：extract_gaps_rag 传入 embed_model
        all_gaps[discipline] = extract_gaps_rag(
            discipline, vectors, chunks, embed_model
        )
        time.sleep(1)

    with open(f"{OUTPUT_DIR}/gaps_raw_rag_fulltext.json", "w",
              encoding="utf-8") as f:
        json.dump(all_gaps, f, indent=2, ensure_ascii=False)

    gap_df, pivot = build_gap_table(all_gaps, OUTPUT_DIR)