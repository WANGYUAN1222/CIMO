"""
CIMO 综合评测脚本（修复版 v3）
修复清单：
  [1] Recall Judge 改为 Blind 评测（随机顺序），消除锚定偏置
  [2] find_paper_text 改用 glob（不回退），避免拿到错误论文
  [3] F1 统一 TP 定义：Precision 阶段复用 Recall 阶段 score=3 的结果
  [4] fuzzy 阈值提高至 0.75，避免错误论文匹配
  [5] Judge 失败率监控，超 5% 自动终止
  [6] processed 去重改用 dedup_key，修复空 folder_name 问题
  [7] 新增 Theme 分层统计（支持实验三）
  [8] 断点续评：评测中断后从已完成条目继续
"""

import json, re, time, os, pathlib, random
import pandas as pd
from difflib import SequenceMatcher
from openai import OpenAI


# ══════════════════════════════════════════════
# 1. 配置
# ══════════════════════════════════════════════
GOLD_CSV        = "./table4_ground_truth_item.csv"
PAPER_FOLDER    = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/Golden_Paper"
MAX_PAPER_CHARS = 8000
FUZZY_THRESHOLD = 0.7        # [修复4] 原 0.6，过低易误匹配
JUDGE_FAIL_CAP  = 0.05        # [修复5] Judge 失败率超过 5% 则终止
JUDGE_FAIL_MIN  = 20          # 至少调用 20 次后才触发失败率检查
SLEEP_SEC       = 1

EVAL_TARGETS = [
    {
        "name"      : "Qwen3-235B",
        "jsonl"     : "./results_qwen3_v5.jsonl",
        "out_recall": "./eval_recall_qwen3_v5.csv",
        "out_prec"  : "./eval_precision_qwen3_v5.csv",
    },
    {
        "name"      : "DeepSeek-V3.2",
        "jsonl"     : "./results_deepseek_v5.jsonl",
        "out_recall": "./eval_recall_deepseek_v5.csv",
        "out_prec"  : "./eval_precision_deepseek_v5.csv",
    },
    {
        "name"      : "LLama3.3-70B",
        "jsonl"     : "./results_llama3.3-70b-base.jsonl",
        "out_recall": "./eval_recall_llama_v5.csv",
        "out_prec"  : "./eval_precision_llama_v5.csv",
    },
    {
        "name"      : "Rule-based",
        "jsonl"     : "./results_rule_based.jsonl",
        "out_recall": "./eval_recall_rulebased_v5.csv",
        "out_prec"  : "./eval_precision_rulebased_v5.csv",
    },
    {
        "name"      : "Random",
        "jsonl"     : "./results_random.jsonl",
        "out_recall": "./eval_recall_random_v5.csv",
        "out_prec"  : "./eval_precision_random_v5.csv",
    }
]

JUDGE_CONFIG = {
    "api_key" : "sk-jMV_sR2kQcV43r9i9SquAQ",
    "base_url": "https://llmapi.paratera.com/v1",
    "model"   : "GLM-4-Plus",
}


# ══════════════════════════════════════════════
# 2. Judge Prompts
# ══════════════════════════════════════════════

# [修复1] 不再区分 Gold/Model，改为比较两个匿名 CIMO
RECALL_JUDGE_SYSTEM = """You are evaluating whether two CIMO instances describe the same research finding.

Evaluate Context (C), Intervention (I), and Outcome (O) INDEPENDENTLY and EQUALLY.

Dimension match criteria:
  C match: Both describe the same or equivalent situational conditions.
           Abstract vs. specific expressions are acceptable
           (e.g., "In strategic alliances" ≈ "In technology outsourcing alliances").
  I match: Both refer to the same type of management action.
           Paraphrasing is acceptable, but the core action must be the same.
           Different mechanisms under the same broad topic do NOT qualify.
  O match: Both outcomes have the same direction and type
           (both positive or both negative). Detail differences are acceptable.

Scoring — count how many dimensions match:
  3 = ALL THREE match    (C ✓, I ✓, O ✓)
  2 = EXACTLY TWO match
  1 = EXACTLY ONE match
  0 = NONE match

Output ONLY JSON:
{"score": <0|1|2|3>, "c_match": <true|false>, "i_match": <true|false>, "o_match": <true|false>, "reason": "one sentence"}"""


PRECISION_JUDGE_SYSTEM = """You are evaluating CIMO extraction quality from supply chain management papers.

Classify each extracted CIMO as exactly ONE of: TP, VN, or FP.
Evaluate Context (C), Intervention (I), and Outcome (O) with EQUAL weight.

TP (True Positive): The extracted CIMO matches a reference CIMO on ALL THREE dimensions (C, I, O).
  Paraphrasing and different abstraction levels are acceptable.

VN (Valid New): NOT in the reference list, AND:
  (a) ALL THREE dimensions (C, I, O) are explicitly supported by the paper excerpt.
  (b) Intervention is a clear management action (not a background condition or factor).
  (c) The C-I-O relationship is coherent and specific.
  When in doubt between VN and FP, choose FP.

FP (False Positive): Any of:
  - Not supported by the paper excerpt
  - Intervention is a condition/factor, not an action
  - C-I-O relationship is incoherent or contradicts the paper

Output ONLY JSON: {"label": "TP", "reason": "one sentence"}
Labels must be exactly: TP, VN, or FP"""


# ══════════════════════════════════════════════
# 3. Judge 调用
# ══════════════════════════════════════════════
def clean_llm_response(raw: str) -> str:
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if '```' in raw:
        for p in raw.split('```'):
            p = p.strip()
            if p.startswith('json'): p = p[4:].strip()
            if p.startswith('{'): return p
    m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
    return m.group(0) if m else raw


def judge_recall(gold: dict, pred: dict, client: OpenAI) -> dict:
    """[修复1] Blind 评测：随机打乱 A/B 顺序，消除锚定偏置"""
    if random.random() > 0.5:
        cimo_a, cimo_b = gold, pred
    else:
        cimo_a, cimo_b = pred, gold

    def get(d, *keys):
        for k in keys:
            if k in d: return d[k]
        return ''

    msg = f"""CIMO_A:
  Context     : {get(cimo_a, 'Context', 'context')}
  Intervention: {get(cimo_a, 'Intervention', 'intervention')}
  Outcome     : {get(cimo_a, 'Outcome', 'outcome')}

CIMO_B:
  Context     : {get(cimo_b, 'Context', 'context')}
  Intervention: {get(cimo_b, 'Intervention', 'intervention')}
  Outcome     : {get(cimo_b, 'Outcome', 'outcome')}

For each dimension, evaluate whether CIMO_A and CIMO_B describe the same thing.
Output JSON: {{"score": <0|1|2|3>, "c_match": <true|false>, "i_match": <true|false>, "o_match": <true|false>, "reason": "..."}}"""

    raw = ''
    try:
        resp = client.chat.completions.create(
            model=JUDGE_CONFIG["model"],
            messages=[
                {"role": "system", "content": RECALL_JUDGE_SYSTEM},
                {"role": "user",   "content": msg},
            ],
            temperature=0, max_tokens=300,
        )
        raw    = resp.choices[0].message.content
        result = json.loads(clean_llm_response(raw))
        score  = int(result.get("score", -1))
        assert score in (0, 1, 2, 3)

        c = bool(result.get("c_match", False))
        i = bool(result.get("i_match", False))
        o = bool(result.get("o_match", False))
        expected = int(c) + int(i) + int(o)
        if score != expected:
            score = expected  # 以布尔值为准自动修正

        return {"score": score, "c_match": c, "i_match": i,
                "o_match": o, "reason": result.get("reason", "")}
    except Exception as e:
        return {"score": -1, "c_match": False, "i_match": False,
                "o_match": False, "reason": f"err:{e}", "raw": raw[:100]}


def judge_precision(pred: dict, gold_items: list,
                    paper_text: str, client: OpenAI) -> dict:
    gold_str = "\n".join([
        f"  [{i+1}] C:{g.get('Context','')} | "
        f"I:{g.get('Intervention','')} | O:{g.get('Outcome','')}"
        for i, g in enumerate(gold_items)
    ]) or "  (none)"

    msg = f"""Extracted CIMO to evaluate:
  Context     : {pred.get('context', '')}
  Intervention: {pred.get('intervention', '')}
  Outcome     : {pred.get('outcome', '')}

Reference CIMOs (C, I, O all equally weighted for TP):
{gold_str}

Paper excerpt (required for VN judgment):
{paper_text[:MAX_PAPER_CHARS]}

Classify as TP, VN, or FP. Output JSON only."""

    raw = ''
    try:
        resp = client.chat.completions.create(
            model=JUDGE_CONFIG["model"],
            messages=[
                {"role": "system", "content": PRECISION_JUDGE_SYSTEM},
                {"role": "user",   "content": msg},
            ],
            temperature=0, max_tokens=250,
        )
        raw    = resp.choices[0].message.content
        result = json.loads(clean_llm_response(raw))
        label  = result.get("label", "").upper().strip()
        assert label in ("TP", "VN", "FP")
        return {"label": label, "reason": result.get("reason", "")}
    except Exception as e:
        return {"label": "ERR", "reason": f"err:{e}", "raw": raw[:100]}


# ══════════════════════════════════════════════
# 4. 工具函数
# ══════════════════════════════════════════════
def norm(t: str) -> str:
    """仅用于标题匹配，转大写+规范化空白"""
    if not isinstance(t, str): return ''
    t = t.upper().strip()
    t = re.sub(r"[''`\u2018\u2019]", "'", t)
    t = re.sub(r'[\"\u201c\u201d]', '"', t)
    return re.sub(r'\s+', ' ', t)


def fuzzy_match(query: str, candidates: list, th: float = FUZZY_THRESHOLD):
    """[修复4] 阈值提高至 0.75"""
    best_score, best = 0.0, None
    for c in candidates:
        s = SequenceMatcher(None, query, c).ratio()
        if s > best_score:
            best_score, best = s, c
    return best if best_score >= th else None


def load_gold(path: str) -> dict:
    df = pd.read_csv(path, encoding='utf-8')
    df['_n'] = df['Title'].apply(norm)
    return {n: grp.to_dict('records') for n, grp in df.groupby('_n')}


def load_preds(path: str) -> dict:
    preds = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            if '_error' in rec: continue
            preds[norm(rec.get('title', ''))] = rec
    return preds


def get_cimo_list(rec: dict) -> list:
    return rec.get('propositions', rec.get('cimo_list', []))


def find_paper_text(folder_name: str) -> str:
    """[修复2] 只在目标目录用 glob 查找，不回退，避免拿到错误论文"""
    if not PAPER_FOLDER or not folder_name or not folder_name.strip():
        return ""
    target = pathlib.Path(PAPER_FOLDER) / folder_name
    if not target.exists():
        print(f"    [警告] 目录不存在: {target}")
        return ""
    mds = list(target.glob("*.md"))   # glob 而非 rglob
    if not mds:
        print(f"    [警告] 未找到 md: {target}")
        return ""
    try:
        return mds[0].read_text(encoding='utf-8')
    except Exception as e:
        print(f"    [警告] 读取失败: {e}")
        return ""


def load_checkpoint(path: str) -> tuple:
    """[修复8] 断点续评：加载已完成的行，返回 (rows, done_keys)"""
    rows, done_keys = [], set()
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')
            rows = df.to_dict('records')
            for row in rows:
                done_keys.add((
                    str(row.get('gold_title', '')),
                    str(row.get('gold_interv', ''))
                ))
            print(f"    [断点续评] 已加载 {len(done_keys)} 条")
        except Exception as e:
            print(f"    [断点续评] 加载失败，从头开始: {e}")
    return rows, done_keys


def check_fail_rate(failed: int, total: int):
    """[修复5] 失败率超阈值时抛出异常"""
    if total >= JUDGE_FAIL_MIN:
        rate = failed / total
        if rate > JUDGE_FAIL_CAP:
            raise RuntimeError(
                f"Judge 失败率过高（{failed}/{total}={rate:.1%} > {JUDGE_FAIL_CAP:.0%}），终止评测"
            )


# ══════════════════════════════════════════════
# 5. 单模型评测
# ══════════════════════════════════════════════
def evaluate_one_model(target: dict, gold: dict, client: OpenAI,
                       max_papers: int = None) -> dict:

    model_name = target["name"]
    out_recall = target["out_recall"]
    out_prec   = target["out_prec"]

    print(f"\n{'='*62}")
    print(f"  评测：{model_name}")
    print(f"{'='*62}")

    preds_map   = load_preds(target["jsonl"])
    gold_titles = list(gold.keys())
    pred_titles = list(preds_map.keys())

    title_map = {
        g: (g if g in preds_map else fuzzy_match(g, pred_titles))
        for g in gold_titles
    }
    if max_papers:
        title_map = dict(list(title_map.items())[:max_papers])
        print(f"[测试模式] 仅处理前 {max_papers} 篇")

    matched_n = sum(1 for v in title_map.values() if v)
    unmatched = [t for t, v in title_map.items() if not v and t.strip()]
    print(f"论文匹配：{matched_n}/{len(gold_titles)}")
    if unmatched:
        print(f"未匹配（{len(unmatched)}篇）：{[t[:40] for t in unmatched]}")

    judge_total  = 0
    judge_failed = 0

    # ── Phase 1: Recall ──────────────────────────────────────────────
    print("\n── Phase 1: Recall（Blind 评测，0/1/2/3）──")

    recall_rows, done_keys = load_checkpoint(out_recall)
    total_gold = sum(len(v) for v in gold.values())
    done_cnt   = len(done_keys)

    for gt, pt in title_map.items():
        gold_items = gold[gt]
        pred_rec   = preds_map.get(pt) if pt else None
        pred_items = get_cimo_list(pred_rec) if pred_rec else []

        for g in gold_items:
            ck = (gt[:80], str(g.get('Intervention', '')))
            if ck in done_keys:
                continue

            done_cnt += 1
            print(f"  [{done_cnt}/{total_gold}] I:{str(g.get('Intervention',''))[:45]}")

            row_base = {
                "gold_title"   : gt[:80],
                "gold_theme"   : g.get('Research Theme', ''),
                "gold_context" : g.get('Context', ''),
                "gold_interv"  : g.get('Intervention', ''),
                "gold_outcome" : g.get('Outcome', ''),
                "pred_count"   : len(pred_items),
            }

            if not pred_items:
                recall_rows.append({**row_base,
                    "best_score": 0, "best_reason": "no pred output",
                    "c_match": False, "i_match": False, "o_match": False,
                    "matched": False, "partial2": False, "partial1": False,
                    "best_pred_ctx": "", "best_pred_int": "", "best_pred_out": "",
                })
                pd.DataFrame(recall_rows).to_csv(out_recall, index=False, encoding='utf-8-sig')
                continue

            best_score, best_res, best_pred = 0, {}, None
            for p in pred_items:
                judge_total += 1
                r = judge_recall(g, p, client)
                time.sleep(SLEEP_SEC)

                if r['score'] == -1:
                    judge_failed += 1
                    print(f"    [Judge失败] {r['reason'][:60]}")
                    check_fail_rate(judge_failed, judge_total)
                    continue

                if r['score'] > best_score:
                    best_score, best_res, best_pred = r['score'], r, p
                if best_score == 3:
                    break

            flag = {3:"✓", 2:"≈", 1:"~"}.get(best_score, "✗")
            print(f"    => {flag} score={best_score}/3  "
                  f"C={'✓' if best_res.get('c_match') else '✗'} "
                  f"I={'✓' if best_res.get('i_match') else '✗'} "
                  f"O={'✓' if best_res.get('o_match') else '✗'}  "
                  f"{best_res.get('reason','')[:40]}")

            recall_rows.append({**row_base,
                "best_score"   : best_score,
                "best_reason"  : best_res.get('reason', ''),
                "c_match"      : best_res.get('c_match', False),
                "i_match"      : best_res.get('i_match', False),
                "o_match"      : best_res.get('o_match', False),
                "matched"      : best_score == 3,
                "partial2"     : best_score == 2,
                "partial1"     : best_score == 1,
                "best_pred_ctx": best_pred.get('context', '')      if best_pred else '',
                "best_pred_int": best_pred.get('intervention', '') if best_pred else '',
                "best_pred_out": best_pred.get('outcome', '')      if best_pred else '',
            })
            # 实时保存（支持断点续评）
            pd.DataFrame(recall_rows).to_csv(out_recall, index=False, encoding='utf-8-sig')

    df_recall = pd.DataFrame(recall_rows)
    df_recall.to_csv(out_recall, index=False, encoding='utf-8-sig')

    # ── Phase 2: Precision ───────────────────────────────────────────
    # [修复3] 收集 Recall 阶段 score=3 的 best_pred_int（归一化）
    #         Precision 阶段遇到这些 Pred 直接标为 TP，避免两套 Judge 不一致
    matched_pred_keys = set()
    for row in recall_rows:
        if row.get('matched') and row.get('best_pred_int'):
            matched_pred_keys.add(norm(str(row['best_pred_int'])))

    print(f"\n── Phase 2: Precision（复用 Recall TP：{len(matched_pred_keys)} 条）──")
    prec_rows = []
    done_p    = 0
    # [修复6] 用 dedup_key 而非 folder_name 去重，修复空字符串问题
    processed = set()

    for gt, pt in title_map.items():
        if not pt or pt not in preds_map:
            continue
        pred_rec    = preds_map[pt]
        folder_name = pred_rec.get('_folder', '').strip()
        dedup_key   = folder_name if folder_name else pt   # 避免空字符串共享
        if dedup_key in processed:
            continue
        processed.add(dedup_key)

        pred_items = get_cimo_list(pred_rec)
        gold_items = gold.get(gt, [])
        paper_text = find_paper_text(folder_name)

        print(f"  {pt[:50]} | pred={len(pred_items)} gold={len(gold_items)}")

        for p in pred_items:
            done_p += 1
            pred_int_key = norm(str(p.get('intervention', '')))

            # [修复3] Recall 阶段已确认为 TP → 直接复用，不重复 Judge
            if pred_int_key in matched_pred_keys:
                r = {"label": "TP", "reason": "confirmed TP in recall phase (score=3)"}
                print(f"    [{done_p}] TP (recall复用)")
            else:
                judge_total += 1
                r = judge_precision(p, gold_items, paper_text, client)
                time.sleep(SLEEP_SEC)
                if r['label'] == 'ERR':
                    judge_failed += 1
                    check_fail_rate(judge_failed, judge_total)
                print(f"    [{done_p}] {r['label']} | {r['reason'][:55]}")

            prec_rows.append({
                "paper_title" : pt[:80],
                "pred_context": p.get('context', ''),
                "pred_interv" : p.get('intervention', ''),
                "pred_outcome": p.get('outcome', ''),
                "gold_count"  : len(gold_items),
                "label"       : r["label"],
                "reason"      : r["reason"],
            })

    df_prec = pd.DataFrame(prec_rows)
    df_prec.to_csv(out_prec, index=False, encoding='utf-8-sig')

    # ── 汇总指标 ─────────────────────────────────────────────────────
    n_gold      = len(df_recall)
    tp_strict   = int(df_recall['matched'].sum())
    tp_lenient2 = int((df_recall['matched'] | df_recall['partial2']).sum())
    tp_lenient1 = int((df_recall['matched'] | df_recall['partial2'] | df_recall['partial1']).sum())

    recall3 = tp_strict   / n_gold if n_gold > 0 else 0.0
    recall2 = tp_lenient2 / n_gold if n_gold > 0 else 0.0
    recall1 = tp_lenient1 / n_gold if n_gold > 0 else 0.0

    recall_c  = df_recall['c_match'].mean()
    recall_i  = df_recall['i_match'].mean()
    recall_o  = df_recall['o_match'].mean()
    avg_score = df_recall['best_score'].mean()

    n_pred        = len(df_prec)
    n_valid       = (df_prec['label'] != 'ERR').sum()
    tp_prec       = (df_prec['label'] == 'TP').sum()
    vn_prec       = (df_prec['label'] == 'VN').sum()
    fp_prec       = (df_prec['label'] == 'FP').sum()
    mod_precision = (tp_prec + vn_prec) / n_valid if n_valid > 0 else 0.0

    def f1(p, r): return 2*p*r/(p+r) if (p+r) > 0 else 0.0
    f1_3 = f1(mod_precision, recall3)
    f1_2 = f1(mod_precision, recall2)
    f1_1 = f1(mod_precision, recall1)

    score_dist = df_recall['best_score'].value_counts().sort_index().to_dict()

    # ── [修复7] Theme 分层统计 ────────────────────────────────────────
    theme_records = []
    print(f"\n── Theme 分层统计 ──")
    if 'gold_theme' in df_recall.columns:
        for theme, grp in df_recall.groupby('gold_theme'):
            r3  = grp['matched'].mean()
            r2  = (grp['matched'] | grp['partial2']).mean()
            r1  = (grp['matched'] | grp['partial2'] | grp['partial1']).mean()
            rc  = grp['c_match'].mean()
            ri  = grp['i_match'].mean()
            ro  = grp['o_match'].mean()
            avg = grp['best_score'].mean()
            theme_records.append({
                "theme": theme, "n_gold": len(grp),
                "recall@3": round(r3, 4), "recall@2": round(r2, 4), "recall@1": round(r1, 4),
                "recall_C": round(rc, 4), "recall_I": round(ri, 4), "recall_O": round(ro, 4),
                "avg_score": round(avg, 4),
            })
            print(f"  {theme[:42]:<42} | n={len(grp):3d} | "
                  f"R@3={r3:.3f} | C={rc:.3f} I={ri:.3f} O={ro:.3f}")

    fail_rate = judge_failed / judge_total if judge_total > 0 else 0.0

    # ── 打印总结 ─────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  {model_name} 评测完成")
    print(f"{'='*62}")
    print(f"  得分分布   : 0={score_dist.get(0,0)} 1={score_dist.get(1,0)} "
          f"2={score_dist.get(2,0)} 3={score_dist.get(3,0)}")
    print(f"  平均得分   : {avg_score:.4f} / 3.0")
    print(f"  Recall@3   : {recall3:.4f}  ({tp_strict}/{n_gold})")
    print(f"  Recall@2   : {recall2:.4f}  ({tp_lenient2}/{n_gold})")
    print(f"  Recall@1   : {recall1:.4f}  ({tp_lenient1}/{n_gold})")
    print(f"  Recall C/I/O: {recall_c:.4f} / {recall_i:.4f} / {recall_o:.4f}")
    print(f"  TP/VN/FP   : {tp_prec}/{vn_prec}/{fp_prec}  "
          f"FP率={fp_prec/n_pred*100:.1f}%" if n_pred > 0 else "")
    print(f"  Mod.Prec   : {mod_precision:.4f}")
    print(f"  F1@3       : {f1_3:.4f}")
    print(f"  F1@2       : {f1_2:.4f}")
    print(f"  F1@1       : {f1_1:.4f}")
    print(f"  Judge失败率 : {judge_failed}/{judge_total} = {fail_rate:.1%}")
    print(f"  输出文件   : {out_recall} / {out_prec}")

    return {
        "model"          : model_name,
        "gold_total"     : n_gold,
        "pred_total"     : n_pred,
        "avg_score"      : round(avg_score, 4),
        "tp_strict"      : tp_strict,
        "tp_lenient2"    : tp_lenient2,
        "tp_lenient1"    : tp_lenient1,
        "recall@3"       : round(recall3, 4),
        "recall@2"       : round(recall2, 4),
        "recall@1"       : round(recall1, 4),
        "recall_C"       : round(recall_c, 4),
        "recall_I"       : round(recall_i, 4),
        "recall_O"       : round(recall_o, 4),
        "pred_TP"        : int(tp_prec),
        "pred_VN"        : int(vn_prec),
        "pred_FP"        : int(fp_prec),
        "mod_precision"  : round(mod_precision, 4),
        "f1@3"           : round(f1_3, 4),
        "f1@2"           : round(f1_2, 4),
        "f1@1"           : round(f1_1, 4),
        "judge_fail_rate": round(fail_rate, 4),
        "score_dist_0"   : score_dist.get(0, 0),
        "score_dist_1"   : score_dist.get(1, 0),
        "score_dist_2"   : score_dist.get(2, 0),
        "score_dist_3"   : score_dist.get(3, 0),
        "_theme_stats"   : theme_records,  # 内部用，不写入汇总 CSV
    }


# ══════════════════════════════════════════════
# 6. 主流程
# ══════════════════════════════════════════════
def run_evaluation(max_papers: int = None):
    gold   = load_gold(GOLD_CSV)
    client = OpenAI(api_key=JUDGE_CONFIG["api_key"],
                    base_url=JUDGE_CONFIG["base_url"])

    all_metrics    = []
    all_theme_rows = []

    for target in EVAL_TARGETS:
        if not os.path.exists(target["jsonl"]):
            print(f"\n[跳过] 文件不存在：{target['jsonl']}")
            continue
        try:
            metrics = evaluate_one_model(target, gold, client, max_papers=max_papers)
            # 拆出 theme_stats，单独汇总
            theme_stats = metrics.pop("_theme_stats", [])
            for ts in theme_stats:
                all_theme_rows.append({"model": metrics["model"], **ts})
            all_metrics.append(metrics)
        except RuntimeError as e:
            print(f"\n[终止] {e}")

    # 总体指标汇总
    if all_metrics:
        df_s = pd.DataFrame(all_metrics).set_index("model")
        print(f"\n{'='*62}\n  对比汇总\n{'='*62}")
        print(df_s.to_string())
        df_s.to_csv("./eval_summary_v3.csv", encoding='utf-8-sig')
        print(f"\n→ ./eval_summary_v3.csv")

    # Theme 分层汇总（支持实验三）
    if all_theme_rows:
        df_theme = pd.DataFrame(all_theme_rows)
        df_theme.to_csv("./eval_theme_breakdown_v2.csv", index=False, encoding='utf-8-sig')
        print(f"→ ./eval_theme_breakdown_v2.csv")


if __name__ == "__main__":
    run_evaluation(max_papers=None)   # 改为 max_papers=5 可先小批次测试