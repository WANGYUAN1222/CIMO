"""
CIMO Batch Extraction Script v5
策略：两步走，互不干扰
  Step 1: 精准抽取 C / I / O（不抽 M，不抽 Theme）
  Step 2: 仅补充 Mechanism (M) 和 Authors
"""

import json
import re
import time
import pathlib
from openai import OpenAI

CONFIGS = {
    "deepseek": {
        "api_key" : "sk-e83c98ffccb645b8b51caec5ca63b717",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model"   : "deepseek-v3.2",
    },
}


# ══════════════════════════════════════════════
# Step 1 Prompt：专注高质量抽取 C / I / O
# ══════════════════════════════════════════════
EXTRACT_SYSTEM = """You are an expert in systematic literature review of supply chain and inter-firm relationship research.
Extract CIO instances (Context / Intervention / Outcome) from academic papers using the framework below.
Do NOT extract Mechanism at this stage.

━━━ DEFINITIONS ━━━

• Context (C): Background conditions the actor FACES. NOT what the actor does.
  Ask: "Under what conditions does this happen?"
  ✅ "In the context of high asset specificity" / "In buyer-supplier relationships"

• Intervention (I): A deliberate ACTION, STRATEGY, or DECISION made by a firm/manager.
  MUST be a gerund phrase starting with an -ing verb.
  Ask: "What does the firm/manager DO?"
  ✅ "choosing hierarchical governance" / "involving suppliers in the NPD project"
  ❌ "higher asset specificity" (condition, not action)

• Outcome (O): The RESULT. Must start with a third-person singular verb (-s/-es).
  Ask: "What happens as a result?"
  ✅ "leads to enhanced performance" / "decreases costs"

━━━ CRITICAL ━━━
WRONG:
  Intervention = "Higher asset specificity"              ← CONDITION, not action
  Outcome      = "Preference for hierarchical governance" ← DECISION, not result
CORRECT:
  Context      = "High asset specificity and behavioral uncertainty"
  Intervention = "Choosing hierarchical governance"
  Outcome      = "leads to enhanced performance"

━━━ OUTPUT RULES ━━━
1. Extract ALL distinct CIO instances (typically 1–5 per paper).
2. Intervention MUST start with an -ing verb.
3. Outcome MUST start with a third-person singular verb (-s/-es).
4. Be specific (15–25 words per field). Do NOT over-abstract.
5. Output ONLY valid JSON. No markdown, no explanation, no <think> blocks.

{
  "title": "<paper title>",
  "cio_list": [
    {
      "id": 1,
      "context": "...",
      "intervention": "...",
      "outcome": "..."
    }
  ]
}"""

EXTRACT_USER = """Extract the paper title and all CIO instances from the paper below.
- Intervention must be an -ing verb phrase.
- Outcome must start with a third-person singular verb (-s/-es).
- Do NOT include Mechanism.
- Output ONLY valid JSON.

Paper:
{paper_text}"""


# ══════════════════════════════════════════════
# Step 2 Prompt：仅补充 Mechanism (M) 和 Authors
# ══════════════════════════════════════════════
ENRICH_SYSTEM = """You are a management research expert.
Given a paper and its extracted CIO instances, your ONLY tasks are:

1. mechanism: For each CIO instance, explain WHY or HOW the intervention produces the outcome.
   - Write 1–2 sentences grounded in the paper's theoretical argument.
   - Use null if the paper does not provide a clear mechanism.

2. authors: Extract author names from the beginning of the paper (comma-separated string).

STRICT RULES:
- DO NOT change context, intervention, or outcome fields.
- DO NOT add any theme or classification.
- Output ONLY valid JSON. No markdown, no explanation, no <think> blocks.

Return format:
{
  "authors": "...",
  "cio_list": [
    {
      "id": <same id>,
      "mechanism": "..." or null
    }
  ]
}"""

ENRICH_USER = """Paper title: {title}

Paper beginning (for author extraction):
{paper_start}

CIO instances to enrich — add mechanism only, DO NOT change context/intervention/outcome:
{cio_json}

Return JSON with authors and mechanism for each id."""


# ══════════════════════════════════════════════
# 文件读取
# ══════════════════════════════════════════════
def read_markdown(md_path: str, max_chars: int = 100_000) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    if len(text) > max_chars:
        print(f"    [截断] {len(text)} → {max_chars} 字符")
        text = text[:max_chars]
    return text.strip()


def find_md_files(root_folder: str):
    """每个子文件夹只取第一个 .md 文件"""
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
    """去除 <think> 块与 Markdown 代码围栏"""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if '```' in raw:
        for part in raw.split('```'):
            part = part.strip()
            if part.startswith('json'):
                part = part[4:].strip()
            if part.startswith('{'):
                return part
    return raw.strip()


# ══════════════════════════════════════════════
# 校验
# ══════════════════════════════════════════════
def validate(cio_list: list):
    bad_starts = ("high", "low", "higher", "lower", "large", "small",
                  "the", "a", "an", "increased", "decreased", "greater", "less")
    for item in cio_list:
        i_words = str(item.get("intervention", "")).lower().split()
        if i_words and i_words[0] in bad_starts:
            print(f"    [警告] Intervention 非动词: {str(item.get('intervention', ''))[:60]}")


# ══════════════════════════════════════════════
# 模型调用
# ══════════════════════════════════════════════
def call_model(messages: list, model_key: str, max_tokens: int = 4000) -> str:
    cfg = CONFIGS[model_key]
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


# ══════════════════════════════════════════════
# Step 1：抽取 C / I / O
# ══════════════════════════════════════════════
def step1_extract(paper_text: str, model_key: str) -> dict:
    raw = call_model([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user",   "content": EXTRACT_USER.format(paper_text=paper_text)},
    ], model_key)
    return json.loads(clean_json(raw))


# ══════════════════════════════════════════════
# Step 2：补充 Mechanism (M) 和 Authors
# ══════════════════════════════════════════════
def step2_enrich(result: dict, paper_text: str, model_key: str) -> dict:
    cio_list = result.get("cio_list", [])
    if not cio_list:
        return result

    raw = call_model([
        {"role": "system", "content": ENRICH_SYSTEM},
        {"role": "user",   "content": ENRICH_USER.format(
            title      = result.get("title", ""),
            paper_start= paper_text[:2000],   # 前 2000 字用于定位作者
            cio_json   = json.dumps(cio_list, ensure_ascii=False, indent=2),
        )},
    ], model_key, max_tokens=3000)

    enriched = json.loads(clean_json(raw))

    # 写入 authors
    result["authors"] = enriched.get("authors", "")

    # 按 id 合并 mechanism，不覆盖 C/I/O
    enriched_map = {
        item.get("id", i + 1): item
        for i, item in enumerate(enriched.get("cio_list", []))
    }
    for idx, c in enumerate(cio_list):
        cid = c.get("id", idx + 1)
        c["mechanism"] = enriched_map.get(cid, {}).get("mechanism", None)

    result["cio_list"] = cio_list
    return result


# ══════════════════════════════════════════════
# 批量抽取主流程
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

            # ── Step 1: 抽取 C / I / O ──
            result = step1_extract(paper_text, model_key)
            n = len(result.get("cio_list", []))
            print(f"    [Step1] 标题: {result.get('title', '')[:60]}")
            print(f"    [Step1] CIO:  {n} 条")
            time.sleep(sleep_sec)


            # ── Step 2: 补充 Mechanism + Authors ──
            result = step2_enrich(result, paper_text, model_key)
            print(f"    [Step2] 作者: {result.get('authors', '')[:60]}")
            if result.get("cio_list"):
                m0 = result["cio_list"][0].get("mechanism", None)
                print(f"    [Step2] mechanism[0]: {str(m0)[:60]}")

            validate(result.get("cio_list", []))

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
    ROOT = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/classified/1"
    #获取所有文件夹
    root = pathlib.Path(ROOT)
    folder_list = list(root.iterdir())
    print(folder_list)
    for folder in folder_list:
        folder_year = pathlib.Path(folder)
        folder_list_year = list(folder_year.iterdir())
        #我只转化2023-2025年数据
        folder_list_year = [f for f in folder_list_year if f.name.startswith("2023") or f.name.endswith("2024") or f.name.endswith("2025")]

        for folder_theme in folder_list_year:
            #获取最后的期刊名和年份
            folder_theme = str(folder_theme)
            journal_name = folder_theme.split("/")[-2]
            year = folder_theme.split("/")[-1]
            print(f"期刊名: {journal_name}, 年份: {year}")
            print(f"./cimo_results/results_deepseek_{journal_name}_{year}.jsonl")
            batch_extract(str(folder_theme), "deepseek", f"./cimo_results_2023_2025/results_deepseek_{journal_name}_{year}.jsonl", sleep_sec=1.0)
