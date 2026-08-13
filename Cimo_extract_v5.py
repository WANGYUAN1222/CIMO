"""
CIMO Batch Extraction Script v5 (修改版)
改进点：
  [1] 双层抽取：每个字段同时输出 specific + abstract 版本
  [2] 去除抽取数量上限，强调完整性优先
  [3] 同一 Intervention 多 Outcome / 多 Context 分开为独立条目
  [4] Step2 Theme 分类增加置信度和 theme_alt
  [5] 校验逻辑增强 + 断点续跑
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

THEME_DESCRIPTION_FOR_PROMPT = "\n".join([
    f"   - {v['name']}\n"
    f"     (Guide: {v['question']} | Keywords: {', '.join(v['keywords'])})"
    for v in THEMES.values()
])


# ══════════════════════════════════════════════
# [修改1+2] Step 1 Prompt：双层抽取 + 无数量上限
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

━━━ CRITICAL DISTINCTION ━━━
WRONG:
  Intervention = "Higher asset specificity"               ← CONDITION, not action
  Outcome      = "Preference for hierarchical governance"  ← DECISION, not result
CORRECT:
  Context      = "High asset specificity and behavioral uncertainty"
  Intervention = "Choosing hierarchical governance"
  Outcome      = "leads to enhanced performance"

━━━ [修改1] DUAL-LEVEL EXPRESSION ━━━
For EACH field, provide TWO versions:
  • _specific : exact language close to the paper (15-25 words)
  • _abstract : synthesized, generalized version at review level (5-12 words)

Example:
  context_specific    : "In buyer-supplier relationships with high asset specificity and volume uncertainty"
  context_abstract    : "In the context of make-or-buy decisions"
  intervention_specific : "maintaining internal capabilities and knowledge about the outsourced technology domain"
  intervention_abstract : "maintaining some knowledge of the outsourced activity"
  outcome_specific    : "significantly increases benefits derived from the outsourcing arrangement"
  outcome_abstract    : "increases outsourcing benefits"

━━━ [修改2] EXTRACTION RULES ━━━
1. Extract ALL distinct CIMO instances. There is NO upper limit on count.
   - Each distinct Intervention = at least one separate instance
   - Same Intervention + multiple Outcomes  → ONE instance PER Outcome
   - Same Intervention + multiple Contexts  → ONE instance PER Context
   - Complex papers typically yield 3-10 instances; do not stop early
   PRIORITY: COMPLETENESS > conciseness. Missing a valid CIMO is worse than
   extracting an extra one.

2. Both _specific and _abstract versions of Intervention MUST start with -ing verb.
3. Both _specific and _abstract versions of Outcome MUST start with -s/-es verb.
4. Output ONLY valid JSON. No markdown, no explanation, no <think> blocks.

━━━ OUTPUT FORMAT ━━━
{
  "title": "<paper title>",
  "cimo_list": [
    {
      "id": 1,
      "context_specific": "...",
      "context_abstract": "...",
      "intervention_specific": "...",
      "intervention_abstract": "...",
      "mechanism": null,
      "outcome_specific": "...",
      "outcome_abstract": "..."
    }
  ]
}"""

EXTRACT_USER = """Extract paper title and ALL CIMO instances from this paper.
Provide both _specific and _abstract versions for each C/I/O field.
Prioritize completeness — extract every distinct finding.
Output ONLY valid JSON.

Paper:
{paper_text}"""


# ══════════════════════════════════════════════
# [修改3] Step 2 Prompt：Theme 置信度 + theme_alt
# ══════════════════════════════════════════════
ENRICH_SYSTEM = f"""You are a management research expert. Given extracted CIMO instances,
add theme classification to each instance.

━━━ THEME DEFINITIONS ━━━
{THEME_DESCRIPTION_FOR_PROMPT}

━━━ [修改3] CLASSIFICATION RULES ━━━
1. Analyze the PRIMARY research question of each CIMO instance, not just keywords.
2. Choose ONE theme per instance that best represents the management problem.
3. Decision shortcuts when uncertain:
   - Governance / contract / outsourcing / make-or-buy        → Theme 1
   - Network positioning / partner selection / tie formation  → Theme 2
   - Trust / commitment / influence / relationship mgmt       → Theme 3
   - Alliance capability / supplier development / integration → Theme 4
   - NPD / innovation / learning / absorptive capacity        → Theme 5
   - EDI / info sharing / coordination / operational KPI      → Theme 6

4. Always output:
   - "theme"            : your best choice (required)
   - "theme_confidence" : "high" | "medium" | "low"
   - "theme_alt"        : alternative theme (only when confidence is medium/low)

Also extract:
- "authors": Author names from the paper beginning (comma-separated).

DO NOT modify: context_specific, context_abstract, intervention_specific,
intervention_abstract, outcome_specific, outcome_abstract.
Output ONLY valid JSON. No markdown, no explanation."""

ENRICH_USER = """Paper title: {title}

Authors section (first 2000 chars of paper):
{paper_start}

CIMO instances to enrich — add theme fields only; DO NOT change C/I/O:
{cimo_json}

Return JSON:
{{
  "authors": "...",
  "cimo_list": [
    {{
      "id": <same as input>,
      "context_specific": "<unchanged>",
      "context_abstract": "<unchanged>",
      "intervention_specific": "<unchanged>",
      "intervention_abstract": "<unchanged>",
      "mechanism": <unchanged or improved>,
      "outcome_specific": "<unchanged>",
      "outcome_abstract": "<unchanged>",
      "theme": "<one of the 6 themes>",
      "theme_alt": "<optional, only when confidence medium/low>",
      "theme_confidence": "high|medium|low"
    }}
  ]
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
# [修改4] 校验：双层字段完整性检查
# ══════════════════════════════════════════════
def validate(cimo_list: list):
    bad_starts = ("high", "low", "higher", "lower", "large", "small",
                  "the", "a", "an", "increased", "decreased", "greater", "less")
    valid_themes = {f"Theme {i}" for i in range(1, 7)}

    for p in cimo_list:
        pid = p.get("id", "?")

        # 检查 abstract 版本起始词
        for field, label in [("intervention_abstract", "I_abs"),
                              ("intervention_specific", "I_spec")]:
            words = str(p.get(field, "")).lower().split()
            if words and words[0] in bad_starts:
                print(f"    [警告] {label} 非动词开头 (id={pid}): "
                      f"{str(p.get(field,''))[:50]}")

        # 检查双层字段是否存在
        required = ["context_abstract", "context_specific",
                    "intervention_abstract", "intervention_specific",
                    "outcome_abstract", "outcome_specific"]
        for field in required:
            if not p.get(field):
                print(f"    [警告] 缺少字段 {field} (id={pid})")

        # 检查 theme
        theme = str(p.get("theme", ""))
        if theme and not any(t in theme for t in valid_themes):
            print(f"    [警告] Theme不合规 (id={pid}): {theme[:50]}")

        # 报告低置信度
        conf = p.get("theme_confidence", "")
        if conf in ("medium", "low"):
            alt = p.get("theme_alt", "")
            print(f"    [注意] Theme置信度={conf} (id={pid}) | "
                  f"主={theme[:30]} | 备={alt[:30]}")


# ══════════════════════════════════════════════
# 模型调用
# ══════════════════════════════════════════════
def call_model(messages: list, model_key: str, max_tokens: int = 6000) -> str:
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
    """Step 1: 双层抽取 C/I/O"""
    raw = call_model([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user",   "content": EXTRACT_USER.format(paper_text=paper_text)},
    ], model_key, max_tokens=6000)
    return json.loads(clean_json(raw))


def step2_enrich(result: dict, paper_text: str, model_key: str) -> dict:
    """Step 2: 补充 Theme（含置信度）/ Mechanism / Authors"""
    cimo_list = result.get("cimo_list", [])
    if not cimo_list:
        return result

    raw = call_model([
        {"role": "system", "content": ENRICH_SYSTEM},
        {"role": "user",   "content": ENRICH_USER.format(
            title      = result.get("title", ""),
            paper_start= paper_text[:2000],
            cimo_json  = json.dumps(cimo_list, ensure_ascii=False, indent=2),
        )},
    ], model_key, max_tokens=4000)

    enriched     = json.loads(clean_json(raw))
    result["authors"] = enriched.get("authors", "")
    enriched_list     = enriched.get("cimo_list", [])
    enriched_map      = {item.get("id", i+1): item
                         for i, item in enumerate(enriched_list)}

    for idx, c in enumerate(cimo_list):
        cid = c.get("id", idx + 1)
        e   = enriched_map.get(cid, {})

        # 只更新 theme 相关字段
        c["theme"]            = e.get("theme", "")
        c["theme_alt"]        = e.get("theme_alt", "")
        c["theme_confidence"] = e.get("theme_confidence", "high")

        # mechanism 只在原来为 null 时补充
        if c.get("mechanism") is None:
            c["mechanism"] = e.get("mechanism", None)

    result["cimo_list"] = cimo_list
    return result


# ══════════════════════════════════════════════
# [修改5] 断点续跑
# ══════════════════════════════════════════════
def load_done_folders(output_path: str) -> set:
    done = set()
    if not pathlib.Path(output_path).exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                folder = rec.get("_folder", "")
                if folder:
                    done.add(folder)
            except Exception:
                pass
    return done


# ══════════════════════════════════════════════
# 批量抽取主流程
# ══════════════════════════════════════════════
def batch_extract(root_folder: str, model_key: str,
                  output_path: str, sleep_sec: float = 3.0):
    md_list = find_md_files(root_folder)
    done    = load_done_folders(output_path)
    todo    = [(f, p) for f, p in md_list if f not in done]

    print(f"找到 {len(md_list)} 篇 | 已完成 {len(done)} 篇 | "
          f"待处理 {len(todo)} 篇 | 模型: {CONFIGS[model_key]['model']}\n")

    with open(output_path, "a", encoding="utf-8") as out_f:
        for i, (folder_name, md_path) in enumerate(todo):
            print(f"[{i+1}/{len(todo)}] {folder_name}")
            try:
                paper_text = read_markdown(str(md_path))

                # Step 1: 双层抽取
                result = step1_extract(paper_text, model_key)
                n = len(result.get("cimo_list", []))
                print(f"    [Step1] 标题 : {result.get('title','')[:60]}")
                print(f"    [Step1] CIMO : {n} 条")
                time.sleep(sleep_sec)

                # Step 2: 补充 Theme/Mechanism/Authors
                result = step2_enrich(result, paper_text, model_key)
                print(f"    [Step2] 作者 : {result.get('authors','')[:55]}")
                if result.get("cimo_list"):
                    c0 = result["cimo_list"][0]
                    print(f"    [Step2] theme    : {c0.get('theme','')[:50]}")
                    print(f"    [Step2] conf     : {c0.get('theme_confidence','')}")
                    print(f"    [Step2] ctx_abs  : {c0.get('context_abstract','')[:50]}")
                    print(f"    [Step2] int_abs  : {c0.get('intervention_abstract','')[:50]}")

                validate(result.get("cimo_list", []))

                result.update({
                    "_folder" : folder_name,
                    "_md_file": md_path.name,
                    "_model"  : CONFIGS[model_key]["model"],
                })
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()

            except Exception as e:
                print(f"    [错误] {e}")
                out_f.write(json.dumps({
                    "_folder": folder_name,
                    "_model" : CONFIGS[model_key]["model"],
                    "_error" : str(e),
                }, ensure_ascii=False) + "\n")
                out_f.flush()

            time.sleep(sleep_sec)

    done_final = load_done_folders(output_path)
    print(f"\n完成！共 {len(done_final)} 篇 → {output_path}")


# ══════════════════════════════════════════════
# 运行入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    ROOT = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/Golden_Paper"

    batch_extract(ROOT, "qwen3",    "./results_qwen3_v6.jsonl",    sleep_sec=1.0)
    batch_extract(ROOT, "deepseek", "./results_deepseek_v6.jsonl", sleep_sec=1.0)

