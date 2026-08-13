"""
Llama 本地推理 CIMO 抽取脚本 v4
=================================
与 Qwen/DeepSeek v4 完全相同的两步策略：
  Step 1: 抽取 C/I/O（v2 风格，高质量）
  Step 2: 补充 Theme / Mechanism / Authors（不改 C/I/O）

硬件：3 × 80G GPU，使用 transformers + device_map=auto
安装：pip install transformers accelerate peft
"""

import json, re, time, pathlib, argparse, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ══════════════════════════════════════════════
# 1. 路径配置 ← 只改这里
# ══════════════════════════════════════════════
UTD24_DIR    = "/data_share_from_3090/wy_code/code/EE/NER/utd24"
PAPER_FOLDER = f"{UTD24_DIR}/output/Golden_Paper"

MODEL_CONFIGS = {
    "llama3.3-70b-base": {
        "model_path"  : f"{UTD24_DIR}/LLM-Research/Llama-3.3-70B-Instruct",
        "lora_path"   : None,
        "output_jsonl": "./results_llama3.3-70b-base_v5.jsonl",
        "n_gpus"      : 3,
    }
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
# 2. Prompts（与 v4 Qwen/DeepSeek 完全相同）
# ══════════════════════════════════════════════

# ── Step 1：抽取 C/I/O ──
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
  Intervention = "Higher asset specificity"              ← CONDITION
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
5. Output ONLY valid JSON. No markdown, no explanation.

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


# ── Step 2：补充 Theme / Mechanism / Authors ──
ENRICH_SYSTEM = f"""You are a management research expert. Given a paper and its extracted CIMO instances,
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
c
2. mechanism: If the existing mechanism field is null, infer the intermediate process
   (10-20 words) explaining WHY the intervention produces the outcome.
   If mechanism already has content, keep it unchanged.
   If truly not inferable, keep null.

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
# 3. 文件工具
# ══════════════════════════════════════════════
def read_markdown(md_path: str, max_chars: int = 60000) -> str:
    with open(md_path, encoding="utf-8") as f:
        text = f.read()
    if len(text) > max_chars:
        text = text[:max_chars]
    return text.strip()

def find_md_files(root_folder: str):
    root = pathlib.Path(root_folder)
    found, seen = [], set()
    for md_path in sorted(root.rglob("*.md")):
        folder = md_path.parent
        if folder in seen: continue
        seen.add(folder)
        found.append((str(folder.relative_to(root)), md_path))
    return found

def clean_json(raw: str) -> str:
    raw = re.sub(r'<\|.*?\|>', '', raw)
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    raw = raw.strip()
    if '```' in raw:
        for p in raw.split('```'):
            p = p.strip()
            if p.startswith('json'): p = p[4:].strip()
            if p.startswith('{'): return p
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    return m.group(0) if m else raw

def validate(cimo_list: list):
    bad = ("high","low","higher","lower","the","a","an","greater","less",
           "increased","decreased","large","small")
    for p in cimo_list:
        w = str(p.get("intervention","")).lower().split()
        if w and w[0] in bad:
            print(f"    [警告] Intervention非动词: {str(p.get('intervention',''))[:55]}")


# ══════════════════════════════════════════════
# 4. 模型加载
# ══════════════════════════════════════════════
def load_model_and_tokenizer(cfg: dict):
    model_path = cfg["model_path"]
    lora_path  = cfg.get("lora_path")

    print(f"加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载模型（device_map=auto）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
    )

    if lora_path:
        print(f"加载 LoRA: {lora_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()

    model.eval()
    print("模型就绪\n")
    return model, tokenizer


# ══════════════════════════════════════════════
# 5. 单次推理（通用）
# ══════════════════════════════════════════════
@torch.no_grad()
def infer(messages: list, model, tokenizer, max_new_tokens: int = 2000) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 28000,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens     = max_new_tokens,
        do_sample          = False,
        repetition_penalty = 1.05,
        pad_token_id       = tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ══════════════════════════════════════════════
# 6. 两步抽取（与 v4 逻辑完全一致）
# ══════════════════════════════════════════════
def step1_extract(paper_text: str, model, tokenizer) -> dict:
    """Step 1: 抽取 C/I/O"""
    raw = infer([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user",   "content": EXTRACT_USER.format(paper_text=paper_text)},
    ], model, tokenizer, max_new_tokens=2000)
    return json.loads(clean_json(raw))


def step2_enrich(result: dict, paper_text: str, model, tokenizer) -> dict:
    """Step 2: 补充 Theme / Mechanism / Authors（不改 C/I/O）"""
    cimo_list = result.get("cimo_list", [])
    if not cimo_list:
        return result

    raw = infer([
        {"role": "system", "content": ENRICH_SYSTEM},
        {"role": "user",   "content": ENRICH_USER.format(
            title      = result.get("title", ""),
            paper_start= paper_text[:2000],
            cimo_json  = json.dumps(cimo_list, ensure_ascii=False, indent=2),
        )},
    ], model, tokenizer, max_new_tokens=1500)

    enriched     = json.loads(clean_json(raw))
    result["authors"] = enriched.get("authors", "")

    enriched_list = enriched.get("cimo_list", [])
    enriched_map  = {item.get("id", i+1): item
                     for i, item in enumerate(enriched_list)}

    for idx, c in enumerate(cimo_list):
        cid = c.get("id", idx+1)
        e   = enriched_map.get(cid, {})
        c["theme"] = e.get("theme", "")
        if c.get("mechanism") is None:
            c["mechanism"] = e.get("mechanism", None)

    result["cimo_list"] = cimo_list
    return result


# ══════════════════════════════════════════════
# 7. 批量抽取（断点续跑）
# ══════════════════════════════════════════════
def batch_extract(model_key: str):
    cfg      = MODEL_CONFIGS[model_key]
    out_path = cfg["output_jsonl"]

    # 读取已完成记录（断点续跑）
    done_folders = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "_folder" in r:
                        done_folders.add(r["_folder"])
                except: pass
        if done_folders:
            print(f"[断点续跑] 已完成 {len(done_folders)} 篇")

    print(f"\n{'='*60}")
    print(f"模型: {model_key}  |  GPU: {cfg['n_gpus']} × 80G")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(cfg)
    md_list = find_md_files(PAPER_FOLDER)
    remain  = [x for x in md_list if x[0] not in done_folders]
    print(f"共 {len(md_list)} 篇，待处理 {len(remain)} 篇\n")

    with open(out_path, "a", encoding="utf-8") as fout:
        for i, (folder_name, md_path) in enumerate(md_list):
            if folder_name in done_folders:
                continue

            print(f"[{i+1}/{len(md_list)}] {folder_name}")
            t0 = time.time()
            try:
                paper_text = read_markdown(str(md_path))

                # Step 1
                result = step1_extract(paper_text, model, tokenizer)
                n = len(result.get("cimo_list", []))
                print(f"    [Step1] {result.get('title','')[:55]}  ({n}条)")

                # Step 2
                result = step2_enrich(result, paper_text, model, tokenizer)
                print(f"    [Step2] 作者: {result.get('authors','')[:50]}")
                if result.get("cimo_list"):
                    print(f"    [Step2] theme[0]: {result['cimo_list'][0].get('theme','')[:50]}")

                validate(result.get("cimo_list", []))
                print(f"    耗时: {time.time()-t0:.0f}s")

                result.update({
                    "_folder" : folder_name,
                    "_md_file": md_path.name,
                    "_model"  : model_key,
                })
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

            except Exception as e:
                print(f"    [错误] {e}")
                fout.write(json.dumps({
                    "_folder": folder_name,
                    "_model" : model_key,
                    "_error" : str(e),
                }, ensure_ascii=False) + "\n")
                fout.flush()

    # 统计
    ok = err = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            (err if "_error" in r else ok).__class__  # 只是占位
            if "_error" in r: err += 1
            else: ok += 1
    print(f"\n完成！成功 {ok} / 失败 {err}  →  {out_path}")


# ══════════════════════════════════════════════
# 8. 入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str,
        default="llama3.3-70b-base",
        choices=list(MODEL_CONFIGS.keys()),
    )
    args = parser.parse_args()
    batch_extract(args.model)