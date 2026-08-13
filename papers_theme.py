import re
import json
import numpy as np
import pathlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
import time

def extract_abstract(md_text: str) -> str:
    PROMPT="""You are an expert academic data parser. I will give you the raw text from the beginning of an academic paper. Your task is to locate and extract the Abstract and Keywords.
    Note: The text might NOT explicitly contain the word 'Abstract'. The abstract is usually the long summary paragraph placed after the author affiliations and before the 'Keywords' or 'Introduction' section.
    Return ONLY valid JSON in this format:
    {{
    "abstract": "...",
    "keywords": ["...", "..."]
    }}
    Raw Text:
    {md_text}"""
    response = client.chat.completions.create(
                model="Qwen3-Next-80B-A3B-Instruct",
                messages=[{"role": "user", "content": PROMPT.format(md_text=md_text)}],
                temperature=0,
                max_tokens=1000
            )
    raw = response.choices[0].message.content.strip()
    print(raw)
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        #只返回abstract字段
        return json.loads(clean)["abstract"]
    except json.JSONDecodeError:
        print("解析失败")
        return ""
    
    


def load_papers_with_abstracts(md_dir: str) -> list:
    """
    加载论文摘要，并与已有CIMO结果对齐
    """
    
    papers = []
    md_files = list(Path(md_dir).rglob("*.md"))
    
    for md_path in md_files:
        paper_id = str(md_path.relative_to(md_dir))
        print(f"处理 {paper_id}...")
        text     = md_path.read_text(encoding="utf-8")
        abstract = extract_abstract(text[:2000])
        
        paper = {
            "paper_id": paper_id,
            "abstract": abstract,
        }
        papers.append(paper)
    
    print(f"共加载 {len(papers)} 篇论文")
    print(f"  有摘要：{sum(1 for p in papers if len(p['abstract'])>100)} 篇")
    return papers


def get_abstract_embeddings(papers: list,
                             embed_model,
                             save_path: str) -> np.ndarray:
    """
    对摘要文本生成embedding
    """
    texts = [p["abstract"] for p in papers]
    
    print(f"生成 {len(texts)} 篇摘要的embedding...")
    embeddings = embed_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    
    np.save(save_path, embeddings)
    print(f"✅ embedding已保存至 {save_path}")
    return embeddings


def find_optimal_k(embeddings: np.ndarray,
                   k_range=range(4, 12)) -> int:
    """
    用轮廓系数确定最优聚类数
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    scores   = {}
    inertias = {}

    for k in k_range:
        km     = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score  = silhouette_score(embeddings, labels)
        scores[k]   = score
        inertias[k] = km.inertia_
        print(f"  K={k}: 轮廓系数={score:.4f}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(list(scores.keys()), list(scores.values()), "bo-")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("轮廓系数（越高越好）")

    axes[1].plot(list(inertias.keys()), list(inertias.values()), "ro-")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Inertia")
    axes[1].set_title("肘部法则")

    plt.tight_layout()
    plt.savefig("./output/optimal_k.png", dpi=150)
    plt.show()

    best_k = max(scores, key=scores.get)
    print(f"\n最优K值：{best_k}（轮廓系数={scores[best_k]:.4f}）")
    return best_k


def run_clustering(embeddings: np.ndarray,
                   papers: list,
                   k: int) -> list:
    """
    执行K-Means聚类，将cluster标签写回papers
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize

    embeddings_norm = normalize(embeddings)
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings_norm)

    for i, paper in enumerate(papers):
        paper["cluster_id"] = int(labels[i])

    # 统计各cluster大小
    from collections import Counter
    sizes = Counter(labels)
    print("\n各Cluster论文数量：")
    for cid, cnt in sorted(sizes.items()):
        print(f"  Cluster {cid}: {cnt} 篇")

    return papers, labels


from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

def extract_cluster_keywords(papers: list,
                              k: int,
                              top_n: int = 10) -> dict:
    """
    为每个cluster提取TF-IDF关键词
    复现Tanskanen Table A1的关键词频率分析
    """
    # 按cluster分组摘要
    cluster_texts = defaultdict(list)
    for paper in papers:
        cid = paper.get("cluster_id", -1)
        cluster_texts[cid].append(paper["abstract"])

    # 全局TF-IDF
    all_texts = [paper["abstract"] for paper in papers]
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2)  # 支持二元词组
    )
    vectorizer.fit(all_texts)
    feature_names = vectorizer.get_feature_names_out()

    cluster_keywords = {}

    for cid in range(k):
        texts = cluster_texts[cid]
        if not texts:
            continue

        # 计算该cluster的TF-IDF矩阵
        tfidf_matrix = vectorizer.transform(texts)
        
        # 均值得分（代表relevance score）
        mean_scores = tfidf_matrix.mean(axis=0).A1
        
        # 词频（count）
        count_matrix = (tfidf_matrix > 0).sum(axis=0).A1

        # 取Top-N关键词
        top_indices = mean_scores.argsort()[::-1][:top_n]
        keywords = []
        for idx in top_indices:
            keywords.append({
                "term":            feature_names[idx],
                "count":           int(count_matrix[idx]),
                "relevance_score": round(
                    mean_scores[idx] / mean_scores[top_indices[0]] * 100, 1
                )
            })

        cluster_keywords[cid] = keywords

    return cluster_keywords


def print_keyword_table(cluster_keywords: dict,
                         cluster_themes: dict):
    """
    打印类似Tanskanen Table A1格式的关键词表
    """
    print("\n" + "="*70)
    print("各主题关键概念（仿Tanskanen Table A1格式）")
    print("="*70)

    for cid, keywords in cluster_keywords.items():
        theme_name = cluster_themes.get(cid, {}).get(
            "theme_name", f"Cluster {cid}"
        )
        n_papers = cluster_themes.get(cid, {}).get("paper_count", 0)
        
        print(f"\nTheme {cid}: {theme_name} (n={n_papers})")
        print(f"  {'关键词':<25} {'Count':>8} {'Relevance':>10}")
        print(f"  {'-'*45}")
        for kw in keywords:
            print(f"  {kw['term']:<25} "
                  f"{kw['count']:>8} "
                  f"{kw['relevance_score']:>9}%")




THEME_NAMING_PROMPT = """
You are an expert in External Resource Management (ERM) research.

Below are key concepts and sample abstracts from a cluster of 
ERM papers published in 2020-2022.

Key concepts (TF-IDF weighted):
{keywords}

Sample abstracts ({n_samples} of {total} papers):
{abstracts}

Task: Identify this cluster's research theme and compare it 
to Tanskanen et al. (2017)'s six ERM themes:
  T1: Decisions on governance mode and mechanisms
  T2: Network formation and relationship initiation
  T3: Inter-organizational relationships
  T4: Strategic aspects of exploiting external resources
  T5: Operational practices of managing external resources
  T6: Learning and innovating with external partners

Return JSON:
{{
  "cluster_id": {cluster_id},
  "theme_name": "concise name (5-8 words)",
  "theme_description": "2-3 sentence description",
  "key_concepts": ["concept1", "concept2", "concept3"],
  "dominant_theory": "main theoretical lens",
  "relation_to_tanskanen": {{
    "type": "continuation/evolution/new_emergence/merger",
    "matched_themes": ["T1", "T2"],
    "explanation": "how it relates or differs"
  }},
  "new_aspects": "what is new compared to 1997-2012 research"
}}
"""

def name_clusters(papers: list,
                   cluster_keywords: dict,
                   k: int,
                   client,
                   output_path: str) -> dict:
    """
    为每个cluster命名主题
    """
    import random
    from collections import defaultdict

    cluster_papers = defaultdict(list)
    for paper in papers:
        cluster_papers[paper["cluster_id"]].append(paper)

    cluster_themes = {}

    for cid in range(k):
        plist  = cluster_papers[cid]
        total  = len(plist)
        sample = random.sample(plist, min(10, total))

        # 格式化关键词
        kw_text = ", ".join([
            f"{kw['term']}({kw['relevance_score']}%)"
            for kw in cluster_keywords.get(cid, [])
        ])

        # 格式化摘要样本
        abs_text = ""
        for i, p in enumerate(sample):
            abs_text += f"\n[{i+1}] {p['abstract'][:300]}...\n"

        prompt = THEME_NAMING_PROMPT.format(
            cluster_id = cid,
            keywords   = kw_text,
            n_samples  = len(sample),
            total      = total,
            abstracts  = abs_text
        )

        try:
            response = client.chat.completions.create(
                model="Qwen3-Next-80B-A3B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1000
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json","").replace("```","").strip()
            theme = json.loads(raw)
            theme["paper_count"] = total
            cluster_themes[cid]  = theme
            print(f"Cluster {cid}: {theme['theme_name']} "
                  f"[{theme['relation_to_tanskanen']['type']}]")
        except Exception as e:
            print(f"Cluster {cid}: 命名失败 - {e}")
            cluster_themes[cid] = {
                "cluster_id":   cid,
                "theme_name":   f"Cluster_{cid}",
                "paper_count":  total
            }
        time.sleep(1)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cluster_themes, f, ensure_ascii=False, indent=2)

    return cluster_themes


def visualize_clusters(embeddings: np.ndarray,
                        labels,
                        cluster_themes: dict,
                        output_path: str):
    """
    UMAP降维可视化，标注主题名称
    """
    import umap
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    reducer      = umap.UMAP(n_components=2, random_state=42)
    embeddings2d = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(14, 10))
    colors  = plt.cm.tab10(np.linspace(0, 1, len(cluster_themes)))

    for cid, color in zip(sorted(cluster_themes.keys()), colors):
        mask = labels == cid
        ax.scatter(
            embeddings2d[mask, 0],
            embeddings2d[mask, 1],
            c=[color], alpha=0.6, s=25, label=f"C{cid}"
        )
        # 标注cluster中心
        cx = embeddings2d[mask, 0].mean()
        cy = embeddings2d[mask, 1].mean()
        name = cluster_themes[cid].get("theme_name", f"C{cid}")
        ax.annotate(
            f"C{cid}: {name[:20]}",
            (cx, cy),
            fontsize=8,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", alpha=0.7)
        )

    ax.set_title("ERM Research Theme Clustering (2020-2022)\nUMAP Visualization")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ 可视化保存至 {output_path}")



import time
from openai import OpenAI

client      = OpenAI(api_key="EMPTY", base_url="http://172.17.65.42:8001/v1")
embed_model = SentenceTransformer("/data_share_from_3090/wy_code/code/EE/NER/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

if __name__ == "__main__":
    # 1. 加载论文摘要
    papers=[]
    path_paper="/data_share_from_3090/wy_code/code/EE/NER/utd24/output/classified/1"
    root = pathlib.Path(path_paper)
    folder_list = list(root.iterdir())
    print(folder_list)
    for folder in folder_list:
        folder_year = pathlib.Path(folder)
        folder_list_year = list(folder_year.iterdir())
        #我只转化2020-2022年数据
        folder_list_year = [f for f in folder_list_year if f.name.startswith("2020") or f.name.endswith("2021") or f.name.endswith("2022")]
        for folder_theme in folder_list_year:
            #获取最后的期刊名和年份
            folder_theme = str(folder_theme)
            paper = load_papers_with_abstracts(
                md_dir     = folder_theme
            )
            # 合并当前期刊的论文摘要
            papers.extend(paper)
    #保存所有论文摘要
    with open("./output_theme/all_papers_abstract.jsonl","w") as f:
        for p in papers:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    # 读取所有论文摘要
    with open("./output_theme/all_papers_abstract.jsonl","r") as f:
        papers = [json.loads(line) for line in f]


    # 2. 生成摘要embedding
    embeddings = get_abstract_embeddings(
        papers     = papers,
        embed_model= embed_model,
        save_path  = "./output_theme/abstract_embeddings_2020_2022.npy"
    )

    # 3. 确定最优K值
    best_k = find_optimal_k(embeddings, k_range=range(4, 12))

    # 4. 执行聚类
    papers, labels = run_clustering(embeddings, papers, k=best_k)

    # 5. 提取关键词（复现Table A1）
    cluster_keywords = extract_cluster_keywords(papers, k=best_k, top_n=10)

    # 6. 命名主题
    cluster_themes = name_clusters(
        papers           = papers,
        cluster_keywords = cluster_keywords,
        k                = best_k,
        client           = client,
        output_path      = "./output/cluster_themes.json"
    )

    # 7. 打印关键词表（Table A1格式）
    print_keyword_table(cluster_keywords, cluster_themes)

    # 8. 可视化
    visualize_clusters(
        embeddings     = embeddings,
        labels         = labels,
        cluster_themes = cluster_themes,
        output_path    = "./output_theme/cluster_visualization.png"
    )

    # # 9. 将cluster标签写回CIMO记录
    # paper_cluster_map = {p["paper_id"]: p["cluster_id"] for p in papers}
    
    # updated_records = []
    # with open("./output_theme/cimo_2020_2022_results.jsonl","r") as f:
    #     for line in f:
    #         r = json.loads(line)
    #         r["cluster_id"] = paper_cluster_map.get(r["paper_id"], -1)
    #         updated_records.append(r)
    
    # with open("./output_theme/cimo_clustered.jsonl","w") as f:
    #     for r in updated_records:
    #         f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n🎉 主题聚类完成，进入Gap识别阶段")