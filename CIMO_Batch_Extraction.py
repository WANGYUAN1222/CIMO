"""
CIMO Batch Extraction Script v4
策略：v2 高质量抽取 + 事后补充 Theme/Mechanism/Authors
两步走，互不干扰：
  Step 1: 用 v2 风格 prompt 精准抽取 C/I/O（不要求抽象化）
  Step 2: 对每篇的抽取结果，再调用一次补充 Theme/Mechanism/Authors
"""

import json
import re
import time
import pathlib
from openai import OpenAI

CONFIGS = {
    "qwen3": {
        "api_key" : "sk-jMV_sR2kQcV43r9i9SquAQ",
        "base_url": "https://llmapi.paratera.com/v1",
        "model"   : "Qwen3-235B-A22B-Instruct-2507",
    },
    "deepseek": {
        "api_key" : "sk-jMV_sR2kQcV43r9i9SquAQ",
        "base_url": "https://llmapi.paratera.com/v1",
        "model"   : "DeepSeek-V3.2",
    },
}

THEMES = {
    "1": {
        "name"    : "Theme 1: Decisions on governance mode and mechanism",
        "question": "How does one select the right governance mode and mechanism?",
        "keywords": ["governance","outsourc","contract","make","buy","hierarchi",
                     "market governance","relational","formal"],
    },
    "2": {
        "name"    : "Theme 2: Network formation and relationship initiation",
        "question": "How does one position the firm in relation to its business environment?",
        "keywords": ["network","tie","alliance formation","partner select",
                     "relationship initiat","weak tie","strong tie","bridging"],
    },
    "3": {
        "name"    : "Theme 3: Interorganizational relationships",
        "question": "How does one manage a relationship with an external actor?",
        "keywords": ["trust","commitment","relational","socialization","dependency",
                     "influence","collaboration","satisfaction","justice","power"],
    },
    "4": {
        "name"    : "Theme 4: Strategic aspects of exploiting external resources",
        "question": "How does one effectively exploit available external resources?",
        "keywords": ["supplier development","alliance learning","integration",
                     "partner complementar","supply network","knowledge"],
    },
    "5": {
        "name"    : "Theme 5: Open innovation and interorganizational learning",
        "question": "How does one learn and innovate with external partners?",
        "keywords": ["npd","new product","innovat","learning","absorptive",
                     "customer involv","supplier involv","r&d","technology"],
    },
    "6": {
        "name"    : "Theme 6: Operational practices of managing external resources",
        "question": "How does one operate with the external resources?",
        "keywords": ["edi","information shar","e-business","integration practice",
                     "communication","coordination","lead-time","agility","quality"],
    },
}

# 动态生成 Prompt 中的 Theme 列表描述
THEME_DESCRIPTION_FOR_PROMPT = "\n".join([
    f"   - {v['name']}\n     (Guide: {v['question']} | Keywords: {', '.join(v['keywords'])})" 
    for v in THEMES.values()
])



# ══════════════════════════════════════════════
# Step 1 Prompt：v2 风格，专注高质量抽取
# ══════════════════════════════════════════════
EXTRACT_SYSTEM = """You are an expert in systematic literature review of supply chain and inter-firm relationship research.
Extract CIMO instances from academic papers using the framework below.

━━━ CIMO DEFINITIONS ━━━

• Context (C): Background conditions the actor FACES. NOT what the actor does.
  Ask: "Under what conditions does this happen?"
  ✅ "In the context of high asset specificity" / "In buyer-supplier relationships"

• Intervention (I): A deliberate ACTION, STRATEGY, or DECISION made by a firm/manager.
  MUST be a gerund phrase starting with an -ing verb.
  Ask: "What does the firm/manager DO?"
  ✅ "choosing hierarchical governance" / "involving suppliers in the NPD project"
  ❌ "higher asset specificity" (condition, not action)

• Mechanism (M): WHY/HOW the intervention produces the outcome. Use null if unclear.

• Outcome (O): The RESULT. Must start with a third-person singular verb (-s/-es).
  Ask: "What happens as a result?"
  ✅ "leads to enhanced performance" / "decreases costs"

━━━ CRITICAL ━━━
WRONG:
  Intervention = "Higher asset specificity"           ← CONDITION
  Outcome      = "Preference for hierarchical governance" ← DECISION, not result
CORRECT:
  Context      = "High asset specificity and behavioral uncertainty"
  Intervention = "Choosing hierarchical governance"
  Outcome      = "leads to enhanced performance"

━━━ OUTPUT RULES ━━━
1. Extract ALL distinct CIMO instances (typically 1–5 per paper).
2. Intervention MUST start with an -ing verb.
3. Outcome MUST start with a third-person singular verb.
4. Be specific (15–25 words per field). Do NOT over-abstract.
5. Output ONLY valid JSON. No markdown, no explanation, no <think> blocks.

{
  "title": "<paper title>",
  "cimo_list": [
    {
      "id": 1,
      "context": "...",
      "intervention": "...",
      "mechanism": null,
      "outcome": "..."
    }
  ]
}"""

EXTRACT_USER = """Extract paper title and all CIMO instances from this paper.
Intervention must be an -ing verb phrase. Outcome must start with -s/-es verb.
Output ONLY valid JSON.

Paper:
{paper_text}"""


# ══════════════════════════════════════════════
# Step 2 Prompt：补充 Theme / Authors（不改 C/I/O）
# ══════════════════════════════════════════════
ENRICH_SYSTEM =f"""You are a management research expert. Given a paper and its extracted CIMO instances,
add two fields to each CIMO:

1. theme: The criteria for topic classification are obtained based on research questions and key words. Please analyze the entire text. Below are the standard definitions for each topic:
    {THEME_DESCRIPTION_FOR_PROMPT},
    Please select one from the following list for output:
    ["Theme 1: Decisions on governance mode and mechanism",
     "Theme 2: Network formation and relationship initiation",
     "Theme 3: Interorganizational relationships",
     "Theme 4: Strategic aspects of exploiting external resources",
     "Theme 5: Open innovation and interorganizational learning",
     "Theme 6: Operational practices of managing external resources"]

Also extract:
3. authors: Author names from the paper beginning (comma-separated).

DO NOT change context, intervention, or outcome fields.
Output ONLY valid JSON. No markdown, no explanation."""

ENRICH_USER = """Paper title: {title}

Authors (extract from paper text if available):
{paper_start}

CIMO instances to enrich (add theme and mechanism; DO NOT change C/I/O):
{cimo_json}

Return JSON:
{{
  "authors": "...",
  "cimo_list": [same array with theme and mechanism added/updated]
}}"""


# ══════════════════════════════════════════════
# 文件读取
# ══════════════════════════════════════════════
def read_markdown(md_path: str, max_chars: int = 100000) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) > max_chars:
        print(f"    [截断] {len(text)} → {max_chars} 字符")
        text = text[:max_chars]
    return text.strip()

def find_md_files(root_folder: str):
    root = pathlib.Path(root_folder)
    found, seen = [], set()
    for md_path in sorted(root.rglob("*.md")):
        folder = md_path.parent
        if folder in seen:
            continue
        seen.add(folder)
        found.append((str(folder.relative_to(root)), md_path))
    return found


# ══════════════════════════════════════════════
# JSON 清理
# ══════════════════════════════════════════════
def clean_json(raw: str) -> str:
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if '```' in raw:
        for p in raw.split('```'):
            p = p.strip()
            if p.startswith('json'): p = p[4:].strip()
            if p.startswith('{'): return p
    return raw.strip()


# ══════════════════════════════════════════════
# 校验
# ══════════════════════════════════════════════
def validate(cimo_list: list):
    bad_starts = ("high","low","higher","lower","large","small",
                  "the","a","an","increased","decreased","greater","less")
    valid_themes = {f"Theme {i}" for i in range(1, 7)}
    for p in cimo_list:
        i_words = str(p.get("intervention","")).lower().split()
        if i_words and i_words[0] in bad_starts:
            print(f"    [警告] Intervention非动词: {str(p.get('intervention',''))[:55]}")
        theme = str(p.get("theme",""))
        if theme and not any(t in theme for t in valid_themes):
            print(f"    [警告] Theme不合规: {theme[:55]}")


# ══════════════════════════════════════════════
# 模型调用
# ══════════════════════════════════════════════
def call_model(messages: list, model_key: str, max_tokens: int = 4000) -> str:
    cfg    = CONFIGS[model_key]
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    kwargs = dict(
        model=cfg["model"],
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    if model_key == "qwen3":
        kwargs["extra_body"] = {"enable_thinking": False}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content.strip()


def step1_extract(paper_text: str, model_key: str) -> dict:
    """Step 1: 抽取 C/I/O（v2 风格，高质量）"""
    raw = call_model([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user",   "content": EXTRACT_USER.format(paper_text=paper_text)},
    ], model_key)
    return json.loads(clean_json(raw))


def step2_enrich(result: dict, paper_text: str, model_key: str) -> dict:
    """Step 2: 补充 Theme / Mechanism / Authors（不改 C/I/O）"""
    cimo_list = result.get("cimo_list", [])
    if not cimo_list:
        return result

    raw = call_model([
        {"role": "system", "content": ENRICH_SYSTEM},
        {"role": "user",   "content": ENRICH_USER.format(
            title      = result.get("title", ""),
            paper_start= paper_text[:2000],   # 只用前2000字找作者
            cimo_json  = json.dumps(cimo_list, ensure_ascii=False, indent=2),
        )},
    ], model_key, max_tokens=3000)

    enriched = json.loads(clean_json(raw))

    # 合并：只取 authors 和 theme/mechanism，其余保持原样
    result["authors"]   = enriched.get("authors", "")
    enriched_list       = enriched.get("cimo_list", [])
    enriched_map        = {item.get("id", i+1): item
                           for i, item in enumerate(enriched_list)}

    for idx, c in enumerate(cimo_list):
        cid = c.get("id", idx+1)
        e   = enriched_map.get(cid, {})
        c["theme"]     = e.get("theme", "")
        # 只在原来是 null 时才覆盖 mechanism
        if c.get("mechanism") is None:
            c["mechanism"] = e.get("mechanism", None)

    result["cimo_list"] = cimo_list
    return result


# ══════════════════════════════════════════════
# 批量抽取
# ══════════════════════════════════════════════
def batch_extract(root_folder: str, model_key: str,
                  output_path: str, sleep_sec: float = 3.0):
    md_list = find_md_files(root_folder)
    print(f"找到 {len(md_list)} 篇 | 模型: {CONFIGS[model_key]['model']}\n")

    results = []
    for i, (folder_name, md_path) in enumerate(md_list):
        print(f"[{i+1}/{len(md_list)}] {folder_name}")
        try:
            paper_text = read_markdown(str(md_path))

            # ── Step 1: 抽取 C/I/O ──
            result = step1_extract(paper_text, model_key)
            n = len(result.get("cimo_list", []))
            print(f"    [Step1] 标题: {result.get('title','')[:60]}")
            print(f"    [Step1] CIMO: {n} 条")
            time.sleep(sleep_sec)

            # ── Step 2: 补充 Theme/Mechanism/Authors ──
            result = step2_enrich(result, paper_text, model_key)
            print(f"    [Step2] 作者: {result.get('authors','')[:55]}")
            if result.get("cimo_list"):
                c0 = result["cimo_list"][0]
                print(f"    [Step2] theme[0]: {c0.get('theme','')[:55]}")

            validate(result.get("cimo_list", []))

            result.update({
                "_folder" : folder_name,
                "_md_file": md_path.name,
                "_model"  : CONFIGS[model_key]["model"],
            })
            results.append(result)

        except Exception as e:
            print(f"    [错误] {e}")
            results.append({
                "_folder": folder_name,
                "_model" : CONFIGS[model_key]["model"],
                "_error" : str(e),
            })
        time.sleep(sleep_sec)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ok  = sum(1 for r in results if "_error" not in r)
    err = sum(1 for r in results if "_error" in r)
    print(f"\n完成！成功 {ok} 篇 / 失败 {err} 篇 → {output_path}")


# ══════════════════════════════════════════════
# 运行入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    ROOT = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/Golden_Paper"

    # 跑 Qwen3
    batch_extract(ROOT, "qwen3", "./results_qwen3_v5.jsonl", sleep_sec=3.0)

    # 跑 DeepSeek（取消注释）
    batch_extract(ROOT, "deepseek", "./results_deepseek_v5.jsonl", sleep_sec=3.0)