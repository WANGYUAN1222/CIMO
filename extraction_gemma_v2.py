"""
Llama 本地推理 CIO 抽取脚本 v5 (Fixed)
=================================
与 Qwen/DeepSeek v5 完全相同的两步策略：
  Step 1: 精准抽取 C/I/O（不抽 M，不抽 Theme）
  Step 2: 仅补充 Mechanism / Authors（不改 C/I/O）

硬件：3 × 80G GPU，使用 transformers + device_map=auto
安装：pip install transformers accelerate peft

修复内容：
  - clean_json 空串/无结构时抛出有意义的 ValueError，不再返回空串
  - step1_extract / step2_enrich 均加 try/except + 原始输出调试打印
  - Step2 解析失败时降级处理（保留 Step1 结果，不整条丢弃）
"""

import json, re, time, pathlib, argparse, os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ══════════════════════════════════════════════
# 1. 路径配置 ← 只改这里
# ══════════════════════════════════════════════
UTD24_DIR    = "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24"
PAPER_FOLDER = f"{UTD24_DIR}/output/Golden_Paper"

MODEL_CONFIGS = {
    "gemma-4": {
        "model_path"  : f"{UTD24_DIR}/gemma-4",
        "lora_path"   : None,
        "output_jsonl": "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24/cimo_results/results_gemma-4_v3.jsonl",
        "n_gpus"      : 4,
    },
}


# ══════════════════════════════════════════════
# 2. Prompts（与 v5 Qwen/DeepSeek 完全相同）
# ══════════════════════════════════════════════

# ── Step 1：专注高质量抽取 C/I/O ──
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
5. Output ONLY valid JSON. No markdown, no explanation, no </tool_call> blocks.

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


# ── Step 2：仅补充 Mechanism (M) 和 Authors ──
ENRICH_SYSTEM = """You are a management research expert.
Given a paper and its extracted CIO instances, your ONLY tasks are:

1. mechanism: For each CIO instance, explain WHY or HOW the intervention produces the outcome.
   - Write 1–2 sentences grounded in the paper's theoretical argument.
   - Use null if the paper does not provide a clear mechanism.

2. authors: Extract author names from the beginning of the paper (comma-separated string).

STRICT RULES:
- DO NOT change context, intervention, or outcome fields.
- DO NOT add any theme or classification.
- Output ONLY valid JSON. No markdown, no explanation, no </tool_call> blocks.

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
# 3. 文件工具
# ══════════════════════════════════════════════
def read_markdown(md_path: str, max_chars: int = 120000) -> str:
    with open(md_path, encoding="utf-8") as f:
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


def clean_json(raw: str) -> str:
    """
    去除特殊标签、</tool_call> 块与 Markdown 代码围栏，支持对象 {} 和数组 []。
    若无法提取任何 JSON 结构，抛出 ValueError（而非返回空串）。
    """
    if not raw or not raw.strip():
        raise ValueError("模型输出为空字符串，无法解析 JSON")

    raw = re.sub(r'<\|.*?\|>', '', raw)
    raw = re.sub(r'</tool_call>.*?</tool_call>', '', raw, flags=re.DOTALL)
    raw = raw.strip()

    if not raw:
        raise ValueError("清理 </tool_call> 块后输出为空，无法解析 JSON")

    # 处理 Markdown 代码围栏
    if '```' in raw:
        for p in raw.split('```'):
            p = p.strip()
            if p.startswith('json'):
                p = p[4:].strip()
            if p.startswith('{') or p.startswith('['):
                return p

    # 优先匹配对象，其次匹配数组
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        return m.group(0)
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        return m.group(0)

    raise ValueError(
        f"clean_json 未找到任何 JSON 结构，原始输出片段：{raw[:300]!r}"
    )


def validate(cio_list: list):
    bad_starts = ("high", "low", "higher", "lower", "large", "small",
                  "the", "a", "an", "increased", "decreased", "greater", "less")
    for item in cio_list:
        i_words = str(item.get("intervention", "")).lower().split()
        if i_words and i_words[0] in bad_starts:
            print(f"    [警告] Intervention 非动词: {str(item.get('intervention', ''))[:60]}")


# ══════════════════════════════════════════════
# 4. 模型加载
# ══════════════════════════════════════════════
def load_model_and_tokenizer(cfg: dict):
    model_path = cfg["model_path"]

    # 使用 AutoTokenizer 而不是 AutoProcessor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 检查并设置聊天模板
    if tokenizer.chat_template is None:
        print("tokenizer.chat_template is None")
        # 为Gemma模型设置适当的聊天模板
        tokenizer.chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
                "{% set sys = messages[0]['content'] %}"
                "{% set messages = messages[1:] %}"
            "{% else %}"
                "{% set sys = '' %}"
            "{% endif %}"
            "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                    "{{ '<start_of_turn>user\n' }}"
                    "{% if loop.first and sys %}{{ sys + '\n\n' }}{% endif %}"
                    "{{ message['content'] + '<end_of_turn>\n<start_of_turn>model\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ message['content'] + '<end_of_turn>\n' }}"
                "{% endif %}"
            "{% endfor %}"
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
    )
    return model, tokenizer


# ══════════════════════════════════════════════
# 5. 单次推理
# ══════════════════════════════════════════════
@torch.no_grad()
def infer(messages: list, model, tokenizer, max_new_tokens: int = 2000) -> str:
    # 移除 enable_thinking 参数，因为它可能导致导入错误
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize              = False,
        add_generation_prompt = True,
        enable_thinking=False
    )
    
    # # 获取prompt的token数量
    # prompt_tokens = tokenizer.encode(prompt)
    # print(f"Input prompt token length: {len(prompt_tokens)}")

    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 28000,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens     = max_new_tokens,
        do_sample          = True,
        temperature        = 0.8,
        top_p              = 0.9,
        repetition_penalty = 1.5,
        pad_token_id       = tokenizer.eos_token_id,
        eos_token_id       = tokenizer.eos_token_id,
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    response = response.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
    return response


# ══════════════════════════════════════════════
# 6. 两步抽取（含完整错误处理）
# ══════════════════════════════════════════════
def step1_extract(paper_text: str, model, tokenizer) -> dict:
    """Step 1: 抽取 C/I/O，解析失败时抛出 ValueError"""
    raw = infer([
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user",   "content": EXTRACT_USER.format(paper_text=paper_text)},
    ], model, tokenizer, max_new_tokens=2000)

    # 调试：打印原始输出前 200 字符，便于定位问题
    print(f"    [Step1 raw 前200字符] {raw[:200]!r}")

    try:
        cleaned = clean_json(raw)
    except ValueError as e:
        raise ValueError(f"Step1 clean_json 失败: {e}")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Step1 JSON 解析失败: {e}\n"
            f"clean_json 输出片段: {cleaned[:300]!r}\n"
            f"原始输出片段: {raw[:300]!r}"
        )

    if isinstance(parsed, list):
        print("    [警告] Step1 返回了列表，自动包装")
        parsed = {"title": "", "cio_list": parsed}

    return parsed


def step2_enrich(result: dict, paper_text: str, model, tokenizer) -> dict:
    """
    Step 2: 补充 Mechanism + Authors（不改 C/I/O）。
    解析失败时降级处理：保留 Step1 结果，打印警告，不抛出异常。
    """
    cio_list = result.get("cio_list", [])
    if not cio_list:
        return result

    raw = infer([
        {"role": "system", "content": ENRICH_SYSTEM},
        {"role": "user",   "content": ENRICH_USER.format(
            title      = result.get("title", ""),
            paper_start= paper_text,
            cio_json   = json.dumps(cio_list, ensure_ascii=False, indent=2),
        )},
    ], model, tokenizer, max_new_tokens=1500)

    # 调试：打印原始输出前 200 字符
    print(f"    [Step2 raw 前200字符] {raw[:200]!r}")

    try:
        cleaned  = clean_json(raw)
        enriched = json.loads(cleaned)
    except (ValueError, json.JSONDecodeError) as e:
        # Step2 失败不影响 Step1 结果落盘，降级处理
        print(f"    [Step2 警告] JSON 解析失败，跳过 mechanism 补充: {e}")
        result["authors"] = ""
        for c in cio_list:
            c.setdefault("mechanism", None)
        result["cio_list"] = cio_list
        return result

    if isinstance(enriched, list):
        print("    [警告] Step2 返回了列表，自动包装")
        enriched = {"authors": "", "cio_list": enriched}

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
# 7. 批量抽取（断点续跑）
# ══════════════════════════════════════════════
def batch_extract(model_key: str):
    cfg      = MODEL_CONFIGS[model_key]
    out_path = cfg["output_jsonl"]

    # 确保输出目录存在
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 读取已完成记录（断点续跑）
    done_folders = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if "_folder" in r:
                        done_folders.add(r["_folder"])
                except Exception:
                    pass
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

                # ── Step 1: 抽取 C/I/O ──
                result = step1_extract(paper_text, model, tokenizer)
                n = len(result.get("cio_list", []))
                print(f"    [Step1] 标题: {result.get('title', '')[:55]}  ({n} 条)")

                # ── Step 2: 补充 Mechanism + Authors ──
                result = step2_enrich(result, paper_text, model, tokenizer)
                print(f"    [Step2] 作者: {result.get('authors', '')[:50]}")
                if result.get("cio_list"):
                    m0 = result["cio_list"][0].get("mechanism", None)
                    print(f"    [Step2] mechanism[0]: {str(m0)[:60]}")

                validate(result.get("cio_list", []))
                print(f"    耗时: {time.time() - t0:.0f}s")

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

    # 最终统计
    ok = err = 0
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if "_error" in r:
                    err += 1
                else:
                    ok += 1
            except Exception:
                pass
    print(f"\n完成！成功 {ok} / 失败 {err}  →  {out_path}")


# ══════════════════════════════════════════════
# 8. 入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str,
        default="gemma-4",
        choices=list(MODEL_CONFIGS.keys()),
    )
    args = parser.parse_args()
    batch_extract(args.model)