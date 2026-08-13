"""
train_cimo_lora_qwen36_27b.py
====================================================================
LoRA 微调 Qwen3.6-27B (dense, bf16 全精度,非 QLoRA)
蒸馏 GPT-5 的 CIMO 抽取(两步法)

★ 模型架构关键点 (Qwen3.6-27B) ★
  - Dense 模型,27B 参数全激活 (对比 35B-A3B 的 3B 激活)
  - 64 层混合架构: 48 × Gated DeltaNet (线性注意力) + 16 × Gated Attention
    每 4 层一组: 3 × (GDN → FFN) + 1 × (Gated Attn → FFN), 共 16 组
  - hidden_size=5120, FFN intermediate=17408
  - 原生多模态(含视觉编码器, 但本任务纯文本, 视觉部分闲置)
  - 原生 262K context (训练用不到)
  - 有 thinking 模式 → 必须禁用,否则破坏 JSON 输出

★ 为什么选 27B 而不是 35B-A3B ★
  1. 知识/推理基准全面超过 35B-A3B (GPQA 87.8 vs 更低, SWE 77.2 vs 73.4)
  2. LoRA 可以加到 FFN, 可训练参数 ~150-250M (vs 35B-A3B 只能加 attention, ~50M)
     FFN 是 transformer 的"知识层", 蒸馏 GPT-5 的领域行为需要它
  3. VRAM 反而更省 (~80GB vs ~95GB), 总参数小
  4. 离线批量处理任务, 不需要 MoE 的推理速度优势

★ VRAM 预估 (bf16 LoRA + FA2 + grad ckpt + seq=14336) ★
  - 模型权重:  ~54 GB (27B × 2)
  - 视觉编码器: ~5 GB
  - Activations: ~15-20 GB
  - LoRA + AdamW8bit (~200M params): ~2-3 GB
  - 合计:       ~80-85 GB
  推荐: 2× 48GB  或  2× 80GB  或  1× H200(141GB)
  你现在的 3 卡布局 (CUDA 0,1,2) 余量充足。

★ 依赖 ★
  pip install -U "transformers>=4.55"   # 必须支持 Qwen3.6 / Gated DeltaNet
  pip install peft accelerate datasets bitsandbytes
  pip install flash-attn --no-build-isolation   # 强烈推荐 (对 16 层标准 attention 生效)

★ 训练-推理对齐 ★
  PAPER_TEXT_MAX_CHARS = 30000  ← 训练时截断长度
  推理脚本 read_markdown(max_chars=30000) 必须改成相同的值!
  推理时也要在 system prompt 加 "/no_think" (本脚本已加)
"""
import os
import json
import re
import glob
import pathlib
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model


# ══════════════════════════════════════════════
# 1. 路径与超参 ← 只改这里
# ══════════════════════════════════════════════
UTD24_DIR       = "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24"
# Qwen3.6-27B 模型路径 (HuggingFace 名: Qwen/Qwen3.6-27B)
# 如果是本地下载, 改成本地绝对路径
MODEL_PATH      = "/public/data_share/model_hub/Qwen3.6-27B"
TRAIN_JSONL_DIR = f"{UTD24_DIR}/2020_2025"
EBM_DIR         = f"{UTD24_DIR}/output/EBM"
OUTPUT_DIR      = f"{UTD24_DIR}/lora_ckpt_qwen/qwen36-27b-cimo-lora-bf16-v1"

# 训练-推理对齐
PAPER_TEXT_MAX_CHARS = 30000
MAX_SEQ_LEN          = 14336

# 训练超参 (dense 27B, LoRA 命中更多模块 ~200M, LR 用 1e-4 较稳)
PER_DEVICE_BATCH     = 1
GRAD_ACCUM           = 8
LEARNING_RATE        = 1e-4
NUM_EPOCHS           = 2
WARMUP_RATIO         = 0.03
LOGGING_STEPS        = 5

# Eval / Save
VAL_RATIO            = 0.10
EVAL_STRATEGY        = "epoch"
SAVE_STRATEGY        = "epoch"
SAVE_TOTAL_LIMIT     = 2
LOAD_BEST_AT_END     = True

# LoRA
LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05

IGNORE_INDEX = -100
SEED         = 42


# ══════════════════════════════════════════════
# 2. Prompts (与推理脚本严格一致)
#    末尾加 /no_think 关闭 Qwen3.6 的 thinking 模式, 保证 JSON 干净
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
}
/no_think"""

EXTRACT_USER = """Extract the paper title and all CIO instances from the paper below.
- Intervention must be an -ing verb phrase.
- Outcome must start with a third-person singular verb (-s/-es).
- Do NOT include Mechanism.
- Output ONLY valid JSON.

Paper:
{paper_text}"""

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
}
/no_think"""

ENRICH_USER = """Paper title: {title}

Paper beginning (for author extraction):
{paper_start}

CIO instances to enrich — add mechanism only, DO NOT change context/intervention/outcome:
{cio_json}

Return JSON with authors and mechanism for each id."""


# ══════════════════════════════════════════════
# 3. 路径解析与样本构造 (与原版完全相同)
# ══════════════════════════════════════════════
def parse_jsonl_filename(fp: str) -> Tuple[Optional[str], Optional[int]]:
    base = os.path.basename(fp).replace(".jsonl", "")
    base = re.sub(r"^results_[^_]+_", "", base)
    m = re.match(r"^(.+)_(\d{4})$", base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def get_paper_id(record: Dict) -> Optional[str]:
    folder = (record.get("_folder") or "").strip()
    if folder:
        return folder.split("/")[0]
    mdfile = (record.get("_md_file") or "").strip()
    if mdfile:
        return os.path.splitext(mdfile)[0]
    return None


def find_md_path(journal: str, year: int, paper_id: str) -> Optional[pathlib.Path]:
    base = pathlib.Path(EBM_DIR) / journal / str(year) / paper_id
    p1 = base / "hybrid_auto" / f"{paper_id}.md"
    if p1.exists():
        return p1
    auto = base / "hybrid_auto"
    if auto.is_dir():
        mds = list(auto.glob("*.md"))
        if mds:
            return mds[0]
    if base.is_dir():
        mds = list(base.rglob("*.md"))
        if mds:
            return mds[0]
    return None


def read_md(p: pathlib.Path, max_chars: int = PAPER_TEXT_MAX_CHARS) -> str:
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars]
    return text.strip()


def build_step1_target(record: Dict) -> str:
    cio = []
    for c in record.get("cio_list", []):
        cio.append({
            "id":           c.get("id"),
            "context":      c.get("context", ""),
            "intervention": c.get("intervention", ""),
            "outcome":      c.get("outcome", ""),
        })
    return json.dumps({"title": record.get("title", ""), "cio_list": cio},
                       ensure_ascii=False, indent=2)


def build_step2_target(record: Dict) -> str:
    cio = []
    for c in record.get("cio_list", []):
        cio.append({"id": c.get("id"), "mechanism": c.get("mechanism", None)})
    return json.dumps({"authors": record.get("authors", ""), "cio_list": cio},
                       ensure_ascii=False, indent=2)


def build_step2_input_cio(record: Dict) -> str:
    cio = []
    for c in record.get("cio_list", []):
        cio.append({
            "id":           c.get("id"),
            "context":      c.get("context", ""),
            "intervention": c.get("intervention", ""),
            "outcome":      c.get("outcome", ""),
        })
    return json.dumps(cio, ensure_ascii=False, indent=2)


def load_training_examples() -> List[Dict]:
    examples = []
    jsonl_files = sorted(glob.glob(os.path.join(TRAIN_JSONL_DIR, "*.jsonl")))
    print(f"\n找到 {len(jsonl_files)} 个 jsonl 文件 in {TRAIN_JSONL_DIR}")

    n_total = n_ok = n_no_md = n_no_cio = n_err = n_no_meta = n_bad_name = 0
    by_journal_year = defaultdict(int)

    for fp in jsonl_files:
        journal, year = parse_jsonl_filename(fp)
        if journal is None:
            print(f"  [skip-bad-name] {os.path.basename(fp)}")
            n_bad_name += 1
            continue

        with open(fp, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    n_err += 1
                    continue
                if "_error" in rec:
                    n_err += 1
                    continue
                if not rec.get("cio_list"):
                    n_no_cio += 1
                    continue

                paper_id = get_paper_id(rec)
                if not paper_id:
                    n_no_meta += 1
                    continue

                md_path = find_md_path(journal, year, paper_id)
                if md_path is None:
                    n_no_md += 1
                    continue

                paper_text = read_md(md_path)
                if not paper_text:
                    n_no_md += 1
                    continue

                paper_uid = f"{journal}/{year}/{paper_id}"

                examples.append({
                    "_folder": paper_uid,
                    "_step":   "step1",
                    "messages": [
                        {"role": "system",    "content": EXTRACT_SYSTEM},
                        {"role": "user",      "content": EXTRACT_USER.format(paper_text=paper_text)},
                        {"role": "assistant", "content": build_step1_target(rec)},
                    ],
                })
                examples.append({
                    "_folder": paper_uid,
                    "_step":   "step2",
                    "messages": [
                        {"role": "system", "content": ENRICH_SYSTEM},
                        {"role": "user",   "content": ENRICH_USER.format(
                            title       = rec.get("title", ""),
                            paper_start = paper_text,
                            cio_json    = build_step2_input_cio(rec),
                        )},
                        {"role": "assistant", "content": build_step2_target(rec)},
                    ],
                })
                n_ok += 1
                by_journal_year[(journal, year)] += 1

    print(f"\n[数据] 总行={n_total}  有效论文={n_ok}")
    print(f"       无md={n_no_md}  无meta={n_no_meta}  无cio={n_no_cio}  错误={n_err}  坏文件名={n_bad_name}")
    print(f"[数据] SFT 样本 = {len(examples)} (每篇 2 条)")

    print("\n[数据] 各 journal/year 论文数:")
    for (j, y), c in sorted(by_journal_year.items()):
        print(f"  {j:45s}  {y}  →  {c} papers")

    return examples


def split_train_val_by_paper(examples: List[Dict],
                              val_ratio: float = VAL_RATIO,
                              seed: int = SEED) -> Tuple[List[Dict], List[Dict]]:
    folders = sorted({ex["_folder"] for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(folders)
    n_val = max(1, int(len(folders) * val_ratio))
    val_folders = set(folders[:n_val])

    train_ex, val_ex = [], []
    for ex in examples:
        (val_ex if ex["_folder"] in val_folders else train_ex).append(ex)

    print(f"\n[split] 论文级 split: total={len(folders)} -> train={len(folders) - n_val}  val={n_val}")
    print(f"[split] 样本数:       train={len(train_ex)}  val={len(val_ex)}")
    return train_ex, val_ex


# ══════════════════════════════════════════════
# 4. Tokenize + 手动构造 labels
#    兼容 Qwen3 系列 enable_thinking 参数, 失败则回退
# ══════════════════════════════════════════════
def apply_template_safe(tokenizer, messages, add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,  # Qwen3 系列特有参数
        )
    except TypeError:
        # 老 tokenizer 不支持该参数, 退回普通调用 (system 里已经写了 /no_think)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def make_tokenize_fn(tokenizer, max_length: int):
    def _tokenize(example):
        messages = example["messages"]
        full_text = apply_template_safe(tokenizer, messages, add_generation_prompt=False)
        full_ids = tokenizer(
            full_text, add_special_tokens=False,
            truncation=True, max_length=max_length,
        )["input_ids"]

        prompt_text = apply_template_safe(tokenizer, messages[:-1], add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_ids)

        labels = list(full_ids)
        for i in range(min(prompt_len, len(labels))):
            labels[i] = IGNORE_INDEX

        return {
            "input_ids":      full_ids,
            "labels":         labels,
            "attention_mask": [1] * len(full_ids),
            "_prompt_len":    prompt_len,
            "_full_len":      len(full_ids),
        }
    return _tokenize


# ══════════════════════════════════════════════
# 5. Pad collator
# ══════════════════════════════════════════════
class PadCollator:
    def __init__(self, pad_token_id: int, label_pad: int = IGNORE_INDEX,
                 pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.label_pad    = label_pad
        self.pad_mult     = pad_to_multiple_of

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_mult:
            max_len = ((max_len + self.pad_mult - 1) // self.pad_mult) * self.pad_mult
        batch = {"input_ids": [], "labels": [], "attention_mask": []}
        for f in features:
            n_pad = max_len - len(f["input_ids"])
            batch["input_ids"].append(f["input_ids"] + [self.pad_token_id] * n_pad)
            batch["labels"].append(f["labels"]    + [self.label_pad]      * n_pad)
            batch["attention_mask"].append(f["attention_mask"] + [0]      * n_pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


# ══════════════════════════════════════════════
# 6. 调试辅助
# ══════════════════════════════════════════════
def print_length_stats(ds: Dataset, name: str, n_sample: int = 200):
    n = min(n_sample, len(ds))
    idxs = random.sample(range(len(ds)), n)
    full_lens   = sorted(ds[i]["_full_len"]   for i in idxs)
    prompt_lens = sorted(ds[i]["_prompt_len"] for i in idxs)

    def pct(arr, p): return arr[min(int(len(arr) * p), len(arr) - 1)]
    print(f"[长度|{name}] full   (n={n}): min={full_lens[0]}  p50={pct(full_lens,0.5)}  "
          f"p90={pct(full_lens,0.9)}  p95={pct(full_lens,0.95)}  p99={pct(full_lens,0.99)}  max={full_lens[-1]}")
    print(f"[长度|{name}] prompt (n={n}): min={prompt_lens[0]}  p50={pct(prompt_lens,0.5)}  "
          f"p90={pct(prompt_lens,0.9)}  max={prompt_lens[-1]}")
    n_trunc = sum(1 for x in full_lens if x >= MAX_SEQ_LEN)
    print(f"[长度|{name}] 触发 max_length={MAX_SEQ_LEN} 截断: {n_trunc}/{n}")


def sanity_check_mask(ds: Dataset, tokenizer, n: int = 2):
    print("\n[sanity] 检查前 2 条样本的 mask 切点:")
    for k in range(min(n, len(ds))):
        ex = ds[k]
        kept = [(i, tid) for i, tid in enumerate(ex["labels"]) if tid != IGNORE_INDEX]
        if not kept:
            print(f"  样本 {k}: ⚠ 没有 token 参与 loss(assistant 段被截没了)")
            continue
        first_idx = kept[0][0]
        tail_prompt = tokenizer.decode(ex["input_ids"][max(0, first_idx - 8):first_idx])
        head_resp   = tokenizer.decode(ex["input_ids"][first_idx:first_idx + 20])
        print(f"  样本 {k}: prompt_len={ex['_prompt_len']}  full_len={ex['_full_len']}  kept_tokens={len(kept)}")
        print(f"           prompt tail: ...{tail_prompt!r}")
        print(f"           assistant head: {head_resp!r}")


# ══════════════════════════════════════════════
# 7. 训练主流程
# ══════════════════════════════════════════════
def main():
    print("=" * 70)
    print("CIMO LoRA Training - Qwen3.6-27B (dense, bf16 全精度)")
    print("  Student: Qwen3.6-27B (dense 27B, 64 layers, hybrid GDN+Attn)")
    print("  Teacher: GPT-5 annotations (results_gpt-5_*.jsonl, 2020-2025)")
    print(f"  PAPER_TEXT_MAX_CHARS={PAPER_TEXT_MAX_CHARS}  MAX_SEQ_LEN={MAX_SEQ_LEN}")
    print("=" * 70)
    print("\n⚠ 重要 1: 推理脚本里 read_markdown() 的 max_chars")
    print(f"   必须改成 {PAPER_TEXT_MAX_CHARS},否则训练-推理输入不一致")
    print("⚠ 重要 2: 推理脚本 system prompt 必须也带 '/no_think',否则会出 <think> 块")
    print("⚠ 重要 3: transformers 必须 >= 4.55 才能加载 Qwen3.6 架构")
    print("=" * 70)

    random.seed(SEED)

    # ---- 1. 数据 ----
    examples = load_training_examples()
    if not examples:
        raise RuntimeError(
            f"没有可用的训练样本。请确认:\n"
            f"  - {TRAIN_JSONL_DIR} 下有 jsonl 文件\n"
            f"  - {EBM_DIR} 下有对应的 <journal>/<year>/<paper_id>/hybrid_auto/<paper_id>.md\n"
        )
    train_examples, val_examples = split_train_val_by_paper(examples)

    # ---- 2. Tokenizer ----
    print("\n加载 tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- 3. Tokenize ----
    print("Tokenizing ...")
    tokenize_fn = make_tokenize_fn(tokenizer, max_length=MAX_SEQ_LEN)
    train_raw = Dataset.from_list(train_examples)
    val_raw   = Dataset.from_list(val_examples)
    train_ds = train_raw.map(tokenize_fn, remove_columns=train_raw.column_names,
                              desc="tokenize train", num_proc=1)
    val_ds   = val_raw.map(tokenize_fn, remove_columns=val_raw.column_names,
                            desc="tokenize val", num_proc=1)
    print_length_stats(train_ds, "train")
    print_length_stats(val_ds,   "val")
    sanity_check_mask(train_ds, tokenizer)
    train_ds = train_ds.remove_columns(["_prompt_len", "_full_len"])
    val_ds   = val_ds.remove_columns(["_prompt_len", "_full_len"])

    # ---- 4. 加载模型 (bf16 全精度) ----
    print("\n加载模型 Qwen3.6-27B (bf16, device_map=auto)...")
    print("  27B 总参 + 视觉编码器 ≈ 59GB, 分到 3 卡, 3-5 分钟")
    print("  注: 视觉编码器加载但不训练 (本任务纯文本)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
        # 强烈建议开 flash-attn 2 (对 16 层标准 Gated Attention 生效)
        # Gated DeltaNet (48 层) 用线性注意力, 不需要 FA2
        # attn_implementation = "flash_attention_2",
    )

    # 加在 model = AutoModelForCausalLM.from_pretrained(...) 后面
    print("\n=== 检查模型模块名 ===")
    import re
    modules = set()
    vision_modules = set()
    for name, _ in model.named_modules():
        # 找带 _proj 的模块
        if re.search(r"_proj$|gate_proj|up_proj|down_proj", name):
            # 拿到模块短名
            short = name.split(".")[-1]
            # 检查是否在视觉部分
            if "visual" in name or "vision" in name or "vit" in name.lower():
                vision_modules.add(short)
            else:
                modules.add(short)

    print(f"主干模块名: {sorted(modules)}")
    print(f"视觉模块名: {sorted(vision_modules)}")
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # ---- 5. LoRA ----
    # Qwen3.6-27B 是 dense 模型, 没有 MoE 专家, 可以放心给所有 linear 层加 LoRA:
    #   - Attention/GDN 投影: q_proj, k_proj, v_proj, o_proj
    #   - FFN (SwiGLU):       gate_proj, up_proj, down_proj
    #
    # 64 层 × 7 模块 × LoRA(r=32) ≈ 150-250M 可训练参数 (~0.5-1% 总参)
    # FFN 是 transformer 存储"领域知识"的主要位置, 蒸馏 GPT-5 必须命中
    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        
        # 只命中标准 Gated Attention 的 o_proj + 全部 FFN
        # 用正则精准排除 GDN 的 q/k/v_proj 和 out_proj
        target_modules = ["o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # 健康检查: 正常输出应该是 trainable params ≈ 150-250M, 占总参 ~0.5-1%
    # 如果远高于这个数(比如几个 G), 说明 target_modules 匹错了非 LM 层

    # ---- 6. Collator ----
    collator = PadCollator(pad_token_id=tokenizer.pad_token_id)

    # ---- 7. TrainingArguments ----
    args = TrainingArguments(
        output_dir                    = OUTPUT_DIR,
        per_device_train_batch_size   = PER_DEVICE_BATCH,
        per_device_eval_batch_size    = PER_DEVICE_BATCH,
        gradient_accumulation_steps   = GRAD_ACCUM,
        num_train_epochs              = NUM_EPOCHS,
        learning_rate                 = LEARNING_RATE,
        lr_scheduler_type             = "cosine",
        warmup_ratio                  = WARMUP_RATIO,
        bf16                          = True,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        logging_steps                 = LOGGING_STEPS,
        eval_strategy                 = EVAL_STRATEGY,
        save_strategy                 = SAVE_STRATEGY,
        save_total_limit              = SAVE_TOTAL_LIMIT,
        load_best_model_at_end        = LOAD_BEST_AT_END,
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        optim                         = "adamw_8bit",
        report_to                     = "none",
        remove_unused_columns         = False,
        label_names                   = ["labels"],
        seed                          = SEED,
        data_seed                     = SEED,
    )

    # ---- 8. Trainer ----
    trainer = Trainer(
        model         = model,
        args          = args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collator,
    )

    print("\n开始训练 ...")
    trainer.train()

    # ---- 9. 保存 ----
    print(f"\n保存 LoRA adapter -> {OUTPUT_DIR}")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n最终 eval 指标:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")

    print(f"\n[推理切换]")
    print(f"  1. 推理脚本 lora_path  = \"{OUTPUT_DIR}\"")
    print(f"  2. 推理脚本 base_model = \"{MODEL_PATH}\"")
    print(f"  3. 推理脚本 read_markdown(max_chars={PAPER_TEXT_MAX_CHARS})  ← 必须改!")
    print(f"  4. 推理脚本 system prompt 末尾加 '/no_think'(本脚本训练时已加)")
    print(f"  5. 推理脚本 PAPER_FOLDER = \"{EBM_DIR}\"")


if __name__ == "__main__":
    main()