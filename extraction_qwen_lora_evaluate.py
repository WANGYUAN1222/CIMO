"""
evaluate_cimo_val_qwen36_27b.py
====================================================================
CIMO LoRA (Qwen3.6-27B) 在测试集(= 训练时的 val_ds 对应论文集)上的评测

★ 设计 ★
- 直接复用训练脚本里所有路径和常量,改一个地方就行(SEED/VAL_RATIO)
- 用同样的种子重现 split_train_val_by_paper,保证测试集论文身份不变
- 对每篇论文跑两步法推理(EXTRACT → ENRICH),与训练对齐
- 用匈牙利算法做 pred ↔ gold 的最优匹配,算 instance-level P/R/F1
- 同时输出 field-level token-F1 / ROUGE-L,看每个字段的质量

★ Qwen3.6 相对 Llama 的关键差异 ★
- system prompt 末尾加 "/no_think",关闭 thinking 模式 (训练时已加)
- apply_chat_template 加 enable_thinking=False (双保险)
- transformers 必须 >= 4.55 (支持 Qwen3.6 / Gated DeltaNet)
- 模型类仍是 AutoModelForCausalLM (官方文档证实)
- 视觉编码器会被加载但不使用,占 ~5GB

★ 用法 ★
  python evaluate_cimo_val_qwen36_27b.py
  python evaluate_cimo_val_qwen36_27b.py --limit 20    # 调试,只跑前 20 篇
  python evaluate_cimo_val_qwen36_27b.py --resume      # 断点续传
  python evaluate_cimo_val_qwen36_27b.py --metrics_only  # 跳过推理,只算指标

★ 输出 ★
  {LORA_PATH}/eval_predictions.jsonl   每篇论文的预测(便于复查)
  {LORA_PATH}/eval_metrics.json        指标汇总
"""

import os
import re
import json
import glob
import time
import math
import pathlib
import random
import argparse
import collections
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


# ══════════════════════════════════════════════
# 1. 配置 ← 直接从训练脚本复制,确保对齐
# ══════════════════════════════════════════════
UTD24_DIR       = "/public/home/lab1/wangyuan_workspace/wy_code/code/EE/NER/utd24"
# Qwen3.6-27B 基座模型路径
MODEL_PATH      = "/public/data_share/model_hub/Qwen3.6-27B"
TRAIN_JSONL_DIR = f"{UTD24_DIR}/2020_2025"
EBM_DIR         = f"{UTD24_DIR}/output/EBM"
LORA_PATH       = f"{UTD24_DIR}/lora_ckpt_qwen/qwen36-27b-cimo-lora-bf16-v1"

# 必须与训练脚本一致
PAPER_TEXT_MAX_CHARS = 30000
VAL_RATIO            = 0.10
SEED                 = 42

# 评测输出
EVAL_DIR             = LORA_PATH
PRED_OUT             = f"{EVAL_DIR}/eval_predictions.jsonl"
METRICS_OUT          = f"{EVAL_DIR}/eval_metrics.json"

# 推理超参
MAX_NEW_TOKENS_STEP1 = 2048
MAX_NEW_TOKENS_STEP2 = 1024
DO_SAMPLE            = False     # 贪心解码,评测可复现

# attn 实现: "sdpa" 最安全(PyTorch 自带),如果 flash-attn 装好了可以改 "flash_attention_2"
ATTN_IMPL            = "sdpa"

# 评测超参
INSTANCE_THRESHOLDS  = [0.3, 0.5, 0.7]   # 三个匹配阈值都看一下
FIELDS_FOR_MATCH     = ["context", "intervention", "outcome"]  # 实例匹配只看 CIO


# ══════════════════════════════════════════════
# 2. Prompts(与训练脚本一字不差, 末尾带 /no_think)
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
# 3. 数据加载(复用训练脚本逻辑,但保留 gold record)
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


def load_all_papers() -> List[Dict]:
    """返回 [{paper_uid, paper_text, gold}, ...],与训练脚本筛选条件完全一致"""
    papers = []
    files = sorted(glob.glob(os.path.join(TRAIN_JSONL_DIR, "*.jsonl")))
    print(f"扫描 {len(files)} 个 jsonl ...")
    for fp in files:
        journal, year = parse_jsonl_filename(fp)
        if journal is None:
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if "_error" in rec or not rec.get("cio_list"):
                    continue
                pid = get_paper_id(rec)
                if not pid:
                    continue
                md = find_md_path(journal, year, pid)
                if md is None:
                    continue
                text = read_md(md)
                if not text:
                    continue
                papers.append({
                    "paper_uid":  f"{journal}/{year}/{pid}",
                    "paper_text": text,
                    "gold":       rec,
                })
    return papers


def split_val_papers(papers: List[Dict],
                      val_ratio: float = VAL_RATIO,
                      seed: int = SEED) -> List[Dict]:
    """与训练脚本 split_train_val_by_paper 完全一致的划分逻辑,只返回 val 部分"""
    uids = sorted({p["paper_uid"] for p in papers})
    rng = random.Random(seed)
    rng.shuffle(uids)
    n_val = max(1, int(len(uids) * val_ratio))
    val_set = set(uids[:n_val])
    val_papers = [p for p in papers if p["paper_uid"] in val_set]
    print(f"[split] total papers = {len(uids)}, val = {len(val_papers)} (seed={seed}, ratio={val_ratio})")
    return val_papers


# ══════════════════════════════════════════════
# 4. 推理:两步法
# ══════════════════════════════════════════════
def apply_template_safe(tokenizer, messages):
    """兼容 Qwen3 系列 enable_thinking 参数 (双保险关 thinking)"""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # 老 tokenizer 不支持 enable_thinking, 退回普通调用 (system 已有 /no_think)
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def strip_think_block(text: str) -> str:
    """如果模型还是输出了 <think>...</think>, 直接剥掉"""
    if not text:
        return text
    # 删除所有 <think>...</think> 段
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 有时候只有开头没有闭合,删除孤立的 <think>... 到第一个 { 之前
    if "<think>" in text.lower() and "</think>" not in text.lower():
        m = re.search(r"\{", text)
        if m:
            text = text[m.start():]
    return text.strip()


def extract_json_from_text(text: str) -> Optional[Dict]:
    """从模型输出抠出最外层 JSON,容错 markdown 代码块 / 前后多余文字 / think 块"""
    if not text:
        return None
    text = strip_think_block(text)
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 括号匹配找最外层 {...}
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens):
    prompt = apply_template_safe(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens = max_new_tokens,
        do_sample      = DO_SAMPLE,
        pad_token_id   = tokenizer.pad_token_id,
        eos_token_id   = tokenizer.eos_token_id,
        use_cache      = True,
    )
    gen = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def predict_one_paper(model, tokenizer, paper_text: str) -> Dict:
    """两步法预测,返回与训练标注同构的 dict(含 title, authors, cio_list with mechanism)"""
    result = {"title": "", "authors": "", "cio_list": [],
              "_step1_raw": "", "_step2_raw": "", "_error": None}

    # ---- Step 1 ----
    msgs1 = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user",   "content": EXTRACT_USER.format(paper_text=paper_text)},
    ]
    out1 = generate(model, tokenizer, msgs1, MAX_NEW_TOKENS_STEP1)
    result["_step1_raw"] = out1
    j1 = extract_json_from_text(out1)
    if not j1 or "cio_list" not in j1:
        result["_error"] = "step1_parse_failed"
        return result

    result["title"] = (j1.get("title") or "").strip()
    raw_list = j1.get("cio_list") or []
    for i, c in enumerate(raw_list, 1):
        if "id" not in c:
            c["id"] = i
    result["cio_list"] = [{
        "id":           c.get("id", i + 1),
        "context":      (c.get("context") or "").strip(),
        "intervention": (c.get("intervention") or "").strip(),
        "outcome":      (c.get("outcome") or "").strip(),
    } for i, c in enumerate(raw_list)]

    if not result["cio_list"]:
        return result

    # ---- Step 2 ----
    cio_in = json.dumps(result["cio_list"], ensure_ascii=False, indent=2)
    msgs2 = [
        {"role": "system", "content": ENRICH_SYSTEM},
        {"role": "user",   "content": ENRICH_USER.format(
            title=result["title"], paper_start=paper_text, cio_json=cio_in)},
    ]
    out2 = generate(model, tokenizer, msgs2, MAX_NEW_TOKENS_STEP2)
    result["_step2_raw"] = out2
    j2 = extract_json_from_text(out2)
    if j2:
        result["authors"] = (j2.get("authors") or "").strip()
        mech_map = {c.get("id"): c.get("mechanism") for c in (j2.get("cio_list") or [])}
        for c in result["cio_list"]:
            c["mechanism"] = mech_map.get(c["id"], None)
    else:
        for c in result["cio_list"]:
            c["mechanism"] = None
    return result


# ══════════════════════════════════════════════
# 5. 评测指标
# ══════════════════════════════════════════════
_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)

def normalize(s: str) -> str:
    return (s or "").lower().strip()


def tokenize(s: str) -> List[str]:
    return _WORD_RE.findall(normalize(s))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style token-level F1"""
    pt = tokenize(pred)
    gt = tokenize(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    common = collections.Counter(pt) & collections.Counter(gt)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pt)
    r = n_common / len(gt)
    return 2 * p * r / (p + r)


def rouge_l(pred: str, gold: str) -> float:
    """ROUGE-L F-measure(基于 LCS)"""
    pt = tokenize(pred)
    gt = tokenize(gold)
    if not pt and not gt:
        return 1.0
    if not pt or not gt:
        return 0.0
    n, m = len(pt), len(gt)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pt[i - 1] == gt[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[n][m]
    if lcs == 0:
        return 0.0
    p = lcs / n
    r = lcs / m
    return 2 * p * r / (p + r)


def instance_similarity(pred_inst: Dict, gold_inst: Dict) -> float:
    """三字段 (C, I, O) token-F1 的平均,用作匹配相似度"""
    scores = [token_f1(pred_inst.get(k, ""), gold_inst.get(k, "")) for k in FIELDS_FOR_MATCH]
    return sum(scores) / len(scores)


def hungarian_match(preds: List[Dict], golds: List[Dict]) -> List[Tuple[int, int, float]]:
    """返回 [(pred_idx, gold_idx, sim), ...] —— 全局最大相似度的一一匹配"""
    if not preds or not golds:
        return []
    n, m = len(preds), len(golds)
    sim = np.zeros((n, m), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            sim[i, j] = instance_similarity(preds[i], golds[j])
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(-sim)
        return [(int(i), int(j), float(sim[i, j])) for i, j in zip(row, col)]
    except ImportError:
        pairs = []
        used_r, used_c = set(), set()
        flat = [(sim[i, j], i, j) for i in range(n) for j in range(m)]
        flat.sort(reverse=True)
        for s, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j)
            pairs.append((i, j, float(s)))
            if len(used_r) == n or len(used_c) == m:
                break
        return pairs


def evaluate_paper(pred: Dict, gold: Dict, threshold: float) -> Dict:
    """单篇论文的 P/R/F1 + 字段细节"""
    preds = pred.get("cio_list", []) or []
    golds = gold.get("cio_list", []) or []

    matches = hungarian_match(preds, golds)
    tp = sum(1 for _, _, s in matches if s >= threshold)
    fp = len(preds) - tp
    fn = len(golds) - tp
    p = tp / len(preds) if preds else (1.0 if not golds else 0.0)
    r = tp / len(golds) if golds else (1.0 if not preds else 0.0)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    field_scores = {f: [] for f in ["context", "intervention", "outcome", "mechanism"]}
    rouge_scores = {f: [] for f in ["context", "intervention", "outcome", "mechanism"]}
    matched_pairs_info = []
    for i, j, s in matches:
        if s < threshold:
            continue
        pi, gj = preds[i], golds[j]
        pair_field = {}
        for field in field_scores:
            p_val = pi.get(field, "") or ""
            g_val = gj.get(field, "") or ""
            if not p_val and not g_val:
                tf = 1.0
                rl = 1.0
            elif not p_val or not g_val:
                tf = 0.0
                rl = 0.0
            else:
                tf = token_f1(p_val, g_val)
                rl = rouge_l(p_val, g_val)
            field_scores[field].append(tf)
            rouge_scores[field].append(rl)
            pair_field[field] = {"token_f1": tf, "rouge_l": rl}
        matched_pairs_info.append({
            "pred_idx": i, "gold_idx": j, "sim": s, "fields": pair_field,
        })

    pred_mech_nonnull = sum(1 for c in preds if c.get("mechanism"))
    gold_mech_nonnull = sum(1 for c in golds if c.get("mechanism"))
    authors_tf = token_f1(pred.get("authors", ""), gold.get("authors", ""))

    return {
        "n_pred": len(preds),
        "n_gold": len(golds),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r, "f1": f1,
        "matches": matched_pairs_info,
        "field_token_f1": {f: (sum(v) / len(v) if v else None)
                            for f, v in field_scores.items()},
        "field_rouge_l":  {f: (sum(v) / len(v) if v else None)
                            for f, v in rouge_scores.items()},
        "pred_mech_nonnull": pred_mech_nonnull,
        "gold_mech_nonnull": gold_mech_nonnull,
        "authors_token_f1":  authors_tf,
    }


def aggregate_metrics(per_paper: List[Dict], threshold: float) -> Dict:
    total_tp = sum(x["tp"] for x in per_paper)
    total_fp = sum(x["fp"] for x in per_paper)
    total_fn = sum(x["fn"] for x in per_paper)

    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0.0

    n = len(per_paper)
    macro_p  = sum(x["precision"] for x in per_paper) / n if n else 0.0
    macro_r  = sum(x["recall"]    for x in per_paper) / n if n else 0.0
    macro_f1 = sum(x["f1"]        for x in per_paper) / n if n else 0.0

    field_tf = {f: [] for f in ["context", "intervention", "outcome", "mechanism"]}
    field_rl = {f: [] for f in ["context", "intervention", "outcome", "mechanism"]}
    for x in per_paper:
        for m in x["matches"]:
            for f, d in m["fields"].items():
                field_tf[f].append(d["token_f1"])
                field_rl[f].append(d["rouge_l"])

    field_summary = {}
    for f in field_tf:
        tf = field_tf[f]; rl = field_rl[f]
        field_summary[f] = {
            "n_pairs": len(tf),
            "mean_token_f1": (sum(tf) / len(tf)) if tf else None,
            "mean_rouge_l":  (sum(rl) / len(rl)) if rl else None,
        }

    authors_tfs = [x["authors_token_f1"] for x in per_paper]
    pred_mech = sum(x["pred_mech_nonnull"] for x in per_paper)
    gold_mech = sum(x["gold_mech_nonnull"] for x in per_paper)
    total_pred = sum(x["n_pred"] for x in per_paper)
    total_gold = sum(x["n_gold"] for x in per_paper)

    return {
        "threshold": threshold,
        "n_papers":  n,
        "instance_level": {
            "total_pred": total_pred,
            "total_gold": total_gold,
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
            "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
            "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        },
        "field_level": field_summary,
        "authors_mean_token_f1": (sum(authors_tfs) / len(authors_tfs)) if authors_tfs else 0.0,
        "mechanism_nonnull_rate": {
            "pred": (pred_mech / total_pred) if total_pred else 0.0,
            "gold": (gold_mech / total_gold) if total_gold else 0.0,
        },
    }


def print_report(metrics_by_thr: Dict, eval_records: List[Dict]):
    print("\n" + "=" * 78)
    print("CIMO LoRA (Qwen3.6-27B) 测试集评测报告")
    print("=" * 78)
    n = metrics_by_thr[INSTANCE_THRESHOLDS[0]]["n_papers"]
    print(f"测试论文数: {n}")
    print(f"已成功推理: {sum(1 for r in eval_records if not r.get('_error'))}")
    print(f"推理失败:   {sum(1 for r in eval_records if r.get('_error'))}")

    print("\n┌─ Instance-level P/R/F1(论文 × 实例级别)")
    print(f"  匹配相似度 = (context, intervention, outcome) token-F1 的平均")
    print(f"  TP = 匹配上且相似度 ≥ τ")
    print(f"  {'τ':<6}{'micro-P':>10}{'micro-R':>10}{'micro-F1':>10}"
          f"{'macro-P':>10}{'macro-R':>10}{'macro-F1':>10}{'TP':>6}{'FP':>6}{'FN':>6}")
    for thr in INSTANCE_THRESHOLDS:
        m = metrics_by_thr[thr]["instance_level"]
        print(f"  {thr:<6.2f}"
              f"{m['micro_precision']*100:>9.2f}%"
              f"{m['micro_recall']*100:>9.2f}%"
              f"{m['micro_f1']*100:>9.2f}%"
              f"{m['macro_precision']*100:>9.2f}%"
              f"{m['macro_recall']*100:>9.2f}%"
              f"{m['macro_f1']*100:>9.2f}%"
              f"{m['tp']:>6}{m['fp']:>6}{m['fn']:>6}")

    print("\n┌─ Field-level(只统计 τ=0.5 下匹配上的 pair)")
    field_m = metrics_by_thr[0.5]["field_level"]
    print(f"  {'field':<14}{'n_pairs':>10}{'mean token-F1':>16}{'mean ROUGE-L':>16}")
    for f in ["context", "intervention", "outcome", "mechanism"]:
        d = field_m[f]
        tf = f"{d['mean_token_f1']*100:.2f}%" if d['mean_token_f1'] is not None else "—"
        rl = f"{d['mean_rouge_l']*100:.2f}%" if d['mean_rouge_l'] is not None else "—"
        print(f"  {f:<14}{d['n_pairs']:>10}{tf:>16}{rl:>16}")

    print("\n┌─ 其它")
    m05 = metrics_by_thr[0.5]
    print(f"  authors 平均 token-F1:        {m05['authors_mean_token_f1']*100:.2f}%")
    print(f"  mechanism 非空率 (pred):      {m05['mechanism_nonnull_rate']['pred']*100:.2f}%")
    print(f"  mechanism 非空率 (gold):      {m05['mechanism_nonnull_rate']['gold']*100:.2f}%")

    total_pred = m05["instance_level"]["total_pred"]
    total_gold = m05["instance_level"]["total_gold"]
    print(f"  总 pred 实例: {total_pred},总 gold 实例: {total_gold},"
          f"  pred/gold = {total_pred/max(total_gold,1):.3f}")
    print("=" * 78)


# ══════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 篇(调试)")
    ap.add_argument("--resume", action="store_true", help="复用已有预测,只补未完成的")
    ap.add_argument("--metrics_only", action="store_true",
                    help="跳过推理,只用已有 predictions 重算指标")
    args = ap.parse_args()

    os.makedirs(EVAL_DIR, exist_ok=True)

    print("=" * 78)
    print("CIMO LoRA Evaluation on val_ds (Qwen3.6-27B)")
    print(f"  base model: {MODEL_PATH}")
    print(f"  LoRA:       {LORA_PATH}")
    print(f"  PAPER_TEXT_MAX_CHARS = {PAPER_TEXT_MAX_CHARS}")
    print(f"  VAL_RATIO = {VAL_RATIO}, SEED = {SEED}")
    print(f"  attn_impl = {ATTN_IMPL}")
    print(f"  pred  → {PRED_OUT}")
    print(f"  stats → {METRICS_OUT}")
    print("=" * 78)
    print("\n⚠ 依赖: transformers >= 4.55 (Qwen3.6 架构), 否则会报 KeyError")
    print("=" * 78)

    # 1. 复现 val 集划分
    all_papers = load_all_papers()
    val_papers = split_val_papers(all_papers)
    val_papers.sort(key=lambda p: p["paper_uid"])  # 固定顺序

    if args.limit and args.limit < len(val_papers):
        val_papers = val_papers[:args.limit]
        print(f"[limit] 截到前 {args.limit} 篇")

    uid2gold = {p["paper_uid"]: p["gold"] for p in val_papers}

    # 2. 推理(可断点续传 / 跳过)
    preds_by_uid: Dict[str, Dict] = {}
    if (args.resume or args.metrics_only) and os.path.exists(PRED_OUT):
        with open(PRED_OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    preds_by_uid[r["paper_uid"]] = r
                except Exception:
                    pass
        print(f"[resume] 已有预测 {len(preds_by_uid)} 篇")

    if not args.metrics_only:
        todo = [p for p in val_papers if p["paper_uid"] not in preds_by_uid]
        print(f"待推理 {len(todo)} 篇")

        if todo:
            print(f"\n加载 base model (bf16, attn_impl={ATTN_IMPL}) ...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base = AutoModelForCausalLM.from_pretrained(
                MODEL_PATH,
                torch_dtype         = torch.bfloat16,
                device_map          = "auto",
                trust_remote_code   = True,
                attn_implementation = ATTN_IMPL,
            )
            print(f"加载 LoRA: {LORA_PATH}")
            model = PeftModel.from_pretrained(base, LORA_PATH)
            model.eval()
            # 推理时关掉 grad ckpt 提速
            if hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            print("模型就绪\n")

            mode = "a" if preds_by_uid else "w"
            t0 = time.time()
            with open(PRED_OUT, mode, encoding="utf-8") as fout:
                for i, p in enumerate(todo, 1):
                    uid = p["paper_uid"]
                    print(f"[{i}/{len(todo)}] {uid}", flush=True)
                    try:
                        pred = predict_one_paper(model, tokenizer, p["paper_text"])
                    except Exception as e:
                        pred = {"_error": f"{type(e).__name__}: {e}",
                                "title": "", "authors": "", "cio_list": []}
                    pred["paper_uid"] = uid
                    preds_by_uid[uid] = pred
                    fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
                    fout.flush()
                    elapsed = time.time() - t0
                    avg = elapsed / i
                    eta = avg * (len(todo) - i)
                    print(f"   抽出 {len(pred.get('cio_list', []))} CIO,"
                          f"avg {avg:.1f}s/篇, ETA {eta/60:.1f}min")

    # 3. 评测
    print("\n开始计算指标 ...")
    per_paper_records = []
    for uid in sorted(uid2gold.keys()):
        gold = uid2gold[uid]
        pred = preds_by_uid.get(uid)
        if pred is None or pred.get("_error"):
            n_gold = len(gold.get("cio_list", []))
            per_paper_records.append({
                "paper_uid": uid,
                "_error": (pred or {}).get("_error", "no_prediction"),
                "n_pred": 0, "n_gold": n_gold,
                "tp": 0, "fp": 0, "fn": n_gold,
                "precision": 0.0, "recall": 0.0, "f1": 0.0,
                "matches": [],
                "field_token_f1": {f: None for f in ["context","intervention","outcome","mechanism"]},
                "field_rouge_l":  {f: None for f in ["context","intervention","outcome","mechanism"]},
                "pred_mech_nonnull": 0,
                "gold_mech_nonnull": sum(1 for c in gold.get("cio_list",[]) if c.get("mechanism")),
                "authors_token_f1": 0.0,
            })
            continue
        records_per_thr = {thr: evaluate_paper(pred, gold, thr) for thr in INSTANCE_THRESHOLDS}
        main_rec = records_per_thr[0.5]
        main_rec["paper_uid"] = uid
        main_rec["thresholds"] = {
            f"{thr:.2f}": {"tp": records_per_thr[thr]["tp"],
                           "fp": records_per_thr[thr]["fp"],
                           "fn": records_per_thr[thr]["fn"],
                           "precision": records_per_thr[thr]["precision"],
                           "recall":    records_per_thr[thr]["recall"],
                           "f1":        records_per_thr[thr]["f1"]}
            for thr in INSTANCE_THRESHOLDS
        }
        per_paper_records.append(main_rec)

    metrics_by_thr = {}
    for thr in INSTANCE_THRESHOLDS:
        per_paper_thr = []
        for uid in sorted(uid2gold.keys()):
            gold = uid2gold[uid]
            pred = preds_by_uid.get(uid)
            if pred is None or pred.get("_error"):
                n_gold = len(gold.get("cio_list", []))
                per_paper_thr.append({
                    "n_pred": 0, "n_gold": n_gold,
                    "tp": 0, "fp": 0, "fn": n_gold,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "matches": [],
                    "field_token_f1": {f: None for f in ["context","intervention","outcome","mechanism"]},
                    "field_rouge_l":  {f: None for f in ["context","intervention","outcome","mechanism"]},
                    "pred_mech_nonnull": 0,
                    "gold_mech_nonnull": sum(1 for c in gold.get("cio_list",[]) if c.get("mechanism")),
                    "authors_token_f1": 0.0,
                })
            else:
                per_paper_thr.append(evaluate_paper(pred, gold, thr))
        metrics_by_thr[thr] = aggregate_metrics(per_paper_thr, thr)

    # 4. 打印 + 保存
    eval_records = [preds_by_uid.get(uid, {"_error": "no_prediction", "paper_uid": uid})
                    for uid in sorted(uid2gold.keys())]
    print_report(metrics_by_thr, eval_records)

    summary = {
        "config": {
            "base_model": MODEL_PATH,
            "lora_path":  LORA_PATH,
            "paper_text_max_chars": PAPER_TEXT_MAX_CHARS,
            "val_ratio": VAL_RATIO, "seed": SEED,
            "thresholds": INSTANCE_THRESHOLDS,
            "match_fields": FIELDS_FOR_MATCH,
            "attn_impl": ATTN_IMPL,
        },
        "n_papers_total": len(uid2gold),
        "n_papers_predicted": sum(1 for r in eval_records if not r.get("_error")),
        "by_threshold": {f"{thr:.2f}": metrics_by_thr[thr] for thr in INSTANCE_THRESHOLDS},
        "per_paper": per_paper_records,
    }
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n指标已保存: {METRICS_OUT}")


if __name__ == "__main__":
    main()