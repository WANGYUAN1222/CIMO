"""
train_cimo_lora.py
====================================================================
LoRA 微调 Llama-3.3-70B-Instruct(bf16 全精度,非 QLoRA),
蒸馏 GPT-5 的 CIMO 抽取(两步法)

★ 数据布局 ★
  训练标注:utd24/2020_2025/results_gpt-5_<journal>_<year>.jsonl  (多个 jsonl)
  论文全文:utd24/output/EBM/<journal>/<year>/<paper_id>/hybrid_auto/<paper_id>.md
  例:
    标注:  2020_2025/results_gpt-5_Academy_of_Management_Journal_2020.jsonl
    全文:  output/EBM/Academy_of_Management_Journal/2020/<paper_id>/hybrid_auto/<paper_id>.md

★ 训练-推理对齐 ★
  PAPER_TEXT_MAX_CHARS = 30000  ← 训练时截断长度
  推理脚本 read_markdown(max_chars=30000) 必须改成相同的值!

依赖:
  pip install "transformers>=4.45" peft accelerate datasets bitsandbytes
  pip install flash-attn --no-build-isolation   # 强烈推荐,显存降 25-30%
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,4,5,6,7"
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
MODEL_PATH      = f"{UTD24_DIR}/LLM-Research/Llama-3.3-70B-Instruct"
TRAIN_JSONL_DIR = f"{UTD24_DIR}/2020_2025"             # 目录,里面有多个 jsonl
EBM_DIR         = f"{UTD24_DIR}/output/EBM"            # 论文全文根目录
OUTPUT_DIR      = f"{UTD24_DIR}/lora_ckpt/llama3.3-70b-cimo-lora-bf16-v1"

# 训练-推理对齐:这两个值必须与推理脚本里的 max_chars 一致
PAPER_TEXT_MAX_CHARS = 30000     # 论文文本字符上限(~8500 tokens)
MAX_SEQ_LEN          = 14336     # 总 token 上限(prompt + assistant)

# 训练超参
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
# 2. Prompts(与推理脚本严格一致)
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
# 3. 路径解析与样本构造
# ══════════════════════════════════════════════
def parse_jsonl_filename(fp: str) -> Tuple[Optional[str], Optional[int]]:
    """
    results_gpt-5_Academy_of_Management_Journal_2020.jsonl
      -> ('Academy_of_Management_Journal', 2020)
    """
    base = os.path.basename(fp).replace(".jsonl", "")
    # 去掉 "results_<model>_" 前缀(model 可能含 '-' 但不含 '_')
    base = re.sub(r"^results_[^_]+_", "", base)
    m = re.match(r"^(.+)_(\d{4})$", base)
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


def get_paper_id(record: Dict) -> Optional[str]:
    """从 record 提取 paper_id。
    优先用 _folder(如 'paper_id' 或 'paper_id/hybrid_auto'),
    退化用 _md_file 的 stem。"""
    folder = (record.get("_folder") or "").strip()
    if folder:
        return folder.split("/")[0]
    mdfile = (record.get("_md_file") or "").strip()
    if mdfile:
        return os.path.splitext(mdfile)[0]
    return None


def find_md_path(journal: str, year: int, paper_id: str) -> Optional[pathlib.Path]:
    """
    查找路径:EBM/<journal>/<year>/<paper_id>/hybrid_auto/<paper_id>.md
    多层 fallback。
    """
    base = pathlib.Path(EBM_DIR) / journal / str(year) / paper_id
    # 标准路径
    p1 = base / "hybrid_auto" / f"{paper_id}.md"
    if p1.exists():
        return p1
    # hybrid_auto 下任意 md
    auto = base / "hybrid_auto"
    if auto.is_dir():
        mds = list(auto.glob("*.md"))
        if mds:
            return mds[0]
    # paper_id 目录下递归找任意 md
    if base.is_dir():
        mds = list(base.rglob("*.md"))
        if mds:
            return mds[0]
    return None


def read_md(p: pathlib.Path, max_chars: int = PAPER_TEXT_MAX_CHARS) -> str:
    """与推理脚本 read_markdown 完全一致的截断逻辑"""
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
    """遍历 TRAIN_JSONL_DIR 下所有 jsonl,关联论文全文,构造 SFT 样本"""
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

                # Step 1 样本
                examples.append({
                    "_folder": paper_uid,
                    "_step":   "step1",
                    "messages": [
                        {"role": "system",    "content": EXTRACT_SYSTEM},
                        {"role": "user",      "content": EXTRACT_USER.format(paper_text=paper_text)},
                        {"role": "assistant", "content": build_step1_target(rec)},
                    ],
                })
                # Step 2 样本
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
    """按 paper_uid 分组 split,同一论文的 Step1/Step2 必须同在 train 或同在 val。"""
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
# ══════════════════════════════════════════════
def make_tokenize_fn(tokenizer, max_length: int):
    def _tokenize(example):
        messages = example["messages"]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        full_ids = tokenizer(
            full_text, add_special_tokens=False,
            truncation=True, max_length=max_length,
        )["input_ids"]

        prompt_text = tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True,
        )
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
    print("CIMO LoRA Training - bf16 全精度(非 QLoRA)")
    print("  Student: Llama-3.3-70B-Instruct")
    print("  Teacher: GPT-5 annotations (results_gpt-5_*.jsonl, 2020-2025)")
    print(f"  PAPER_TEXT_MAX_CHARS={PAPER_TEXT_MAX_CHARS}  MAX_SEQ_LEN={MAX_SEQ_LEN}")
    print("=" * 70)
    print("\n⚠ 重要:推理脚本里 read_markdown() 的 max_chars")
    print(f"   必须改成 {PAPER_TEXT_MAX_CHARS},否则训练-推理输入不一致")
    print("=" * 70)

    random.seed(SEED)

    # ---- 1. 数据 ----
    examples = load_training_examples()
    if not examples:
        raise RuntimeError(
            f"没有可用的训练样本。请确认:\n"
            f"  - {TRAIN_JSONL_DIR} 下有 jsonl 文件\n"
            f"  - {EBM_DIR} 下有对应的 <journal>/<year>/<paper_id>/hybrid_auto/<paper_id>.md\n"
            f"  - 上方 [数据] 日志中 '无md' 计数不应过高"
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

    # ---- 4. 加载模型 (bf16 全精度,非 QLoRA)----
    print("\n加载模型 (bf16 全精度, device_map=auto, 70B≈140GB 分到 3 卡)...")
    print("  这一步比较慢,5-10 分钟,请耐心等待")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype       = torch.bfloat16,
        device_map        = "auto",
        trust_remote_code = True,
        # 如果装了 flash-attn,取消下面注释,显存降 25-30%
        # attn_implementation = "flash_attention_2",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # ---- 5. LoRA ----
    lora_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        # 纯 LoRA 用 adamw_8bit:模型 bf16 不量化,优化器状态 8bit 省显存
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
    print(f"  1. 推理脚本 lora_path = \"{OUTPUT_DIR}\"")
    print(f"  2. 推理脚本 read_markdown(max_chars={PAPER_TEXT_MAX_CHARS})  ← 必须改!")
    print(f"  3. 推理脚本 PAPER_FOLDER = \"{EBM_DIR}\"(如果原本指向 Golden_Paper,要改)")


if __name__ == "__main__":
    main()