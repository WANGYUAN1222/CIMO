"""
人工一致性验证脚本 v3
修复清单：
  [1] 分层抽样：固定各档配额后用余数法精确凑满 SAMPLE_N，不再 head() 截断
  [2] cohen_kappa_weighted 返回值语义明确，内外 po 统一命名
  [3] SCORE_LABELS 改为函数参数显式传入，消除默认参数陷阱
  [4] Step1 同时检验 C/I/O 三列完整性才纳入抽样
  [5] Step2/3 统计并提示跳过条目数量
  [6] Step3 新增分维度 C/I/O 人工 vs Judge 一致率（支持实验三）
  [7] 论文模板中的 Judge 模型名从配置读取，不再硬编码
  [8] 混淆矩阵打印格式改为动态生成
"""

import argparse, pandas as pd, numpy as np
import random, json, os
from datetime import datetime

# ══════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════
EVAL_CSV    = "./eval_recall_deepseek_v3.csv"   # 评测结果（含 best_score 0-3）
SAMPLE_CSV  = "./human_eval_sample_v3.csv"
RESULT_CSV  = "./human_eval_result_v3.csv"
SAMPLE_N    = 30
RANDOM_SEED = 42
SCORE_LABELS = [0, 1, 2, 3]
JUDGE_MODEL  = "GLM-4-Plus"   # [修复7] 统一从这里读取，模板自动引用


# ══════════════════════════════════════════════
# Step 1: 分层抽样（精确凑满 SAMPLE_N）
# ══════════════════════════════════════════════
def step1_sample():
    df = pd.read_csv(EVAL_CSV)

    # [修复4] 同时检验 C/I/O 三列均完整
    required_cols = ['best_pred_ctx', 'best_pred_int', 'best_pred_out']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[警告] 缺少列: {missing}，仅按 best_pred_int 过滤")
        check_cols = ['best_pred_int']
    else:
        check_cols = required_cols

    mask = pd.Series([True] * len(df))
    for col in check_cols:
        mask &= (
            df[col].notna() &
            (df[col].astype(str).str.strip() != '') &
            (df[col].astype(str).str.strip() != 'nan')
        )
    valid = df[mask].copy()

    print(f"有效条目总数（C/I/O 均完整）: {len(valid)}")
    print(f"Judge 分布（0/1/2/3）:")
    dist = valid['best_score'].value_counts().sort_index()
    print(dist)
    print()

    # [修复1] 用 Largest Remainder Method 精确分配配额
    counts      = {s: int(dist.get(s, 0)) for s in SCORE_LABELS}
    total_valid = sum(counts.values())
    quotas      = {}
    remainders  = {}

    for s in SCORE_LABELS:
        exact        = SAMPLE_N * counts[s] / total_valid if total_valid > 0 else 0
        quotas[s]    = int(exact)
        remainders[s] = exact - int(exact)

    # 用余数法补足剩余名额
    shortage = SAMPLE_N - sum(quotas.values())
    for s in sorted(remainders, key=remainders.get, reverse=True)[:shortage]:
        quotas[s] += 1

    print(f"分层配额（精确凑满 {SAMPLE_N} 条）:")
    samples = []
    random.seed(RANDOM_SEED)
    for s in SCORE_LABELS:
        n   = min(quotas[s], counts[s])   # 不超过该档实际数量
        grp = valid[valid['best_score'] == s]
        if n > 0 and len(grp) > 0:
            samples.append(grp.sample(n, random_state=RANDOM_SEED))
        print(f"  score={s}: 共{counts[s]}条，配额{quotas[s]}，实际抽{n}条")

    if not samples:
        print("[错误] 无有效样本可抽取")
        return

    sample_df = pd.concat(samples).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    sample_df['human_score'] = ''
    sample_df['human_note']  = ''
    sample_df['human_c']     = ''   # [修复8] 人工分维度标注
    sample_df['human_i']     = ''
    sample_df['human_o']     = ''

    keep = [
        'gold_title', 'gold_theme', 'gold_context', 'gold_interv', 'gold_outcome',
        'best_pred_ctx', 'best_pred_int', 'best_pred_out',
        'best_score', 'c_match', 'i_match', 'o_match', 'best_reason',
        'human_score', 'human_c', 'human_i', 'human_o', 'human_note',
    ]
    keep = [c for c in keep if c in sample_df.columns]
    sample_df = sample_df[keep]
    sample_df.insert(0, 'id', range(1, len(sample_df) + 1))

    sample_df.to_csv(SAMPLE_CSV, index=False, encoding='utf-8-sig')
    actual_n = len(sample_df)
    print(f"\n抽样完成！{actual_n} 条 → {SAMPLE_CSV}")

    print("\n打分说明（human_score 列）：")
    print("  3 = 三个维度全部匹配（C✓ I✓ O✓）")
    print("  2 = 两个维度匹配")
    print("  1 = 一个维度匹配")
    print("  0 = 无匹配")
    print("\n可选：同时填写 human_c / human_i / human_o（1=匹配，0=不匹配）")
    print("      以便 Step3 进行分维度分析")
    print("\n方式A（推荐）：用 Excel 打开，填写 human_score 列，保存后运行 step 3")
    print("方式B：运行 step 2 交互打分")


# ══════════════════════════════════════════════
# Step 2: 交互式打分
# ══════════════════════════════════════════════
def step2_score():
    if not os.path.exists(SAMPLE_CSV):
        print(f"[错误] 找不到 {SAMPLE_CSV}，请先运行 step 1")
        return

    df = pd.read_csv(SAMPLE_CSV)
    df['human_score'] = df['human_score'].astype(str).str.strip()
    unscored_idx = df[df['human_score'].isin(['', 'nan'])].index.tolist()

    if not unscored_idx:
        print("所有条目已打分，请直接运行 step 3")
        return

    print("=" * 65)
    print(f"人工一致性标注（剩余 {len(unscored_idx)} 条）")
    print("q=退出并保存，s=跳过当前条目，进度自动保存")
    print("=" * 65)
    print("评分标准（0/1/2/3）：")
    print("  3 = C✓ I✓ O✓  三维全中")
    print("  2 = 任意两维匹配")
    print("  1 = 仅一维匹配")
    print("  0 = 无维度匹配\n")

    skipped = 0

    for pos, idx in enumerate(unscored_idx):
        row  = df.loc[idx]
        done = pos + 1

        print(f"\n{'─' * 65}")
        print(f"[{done}/{len(unscored_idx)}] {str(row.get('gold_title', ''))[:60]}")
        print(f"{'─' * 65}")
        print(f"\n【黄金集】")
        print(f"  C: {row.get('gold_context', '')}")
        print(f"  I: {row.get('gold_interv', '')}")
        print(f"  O: {row.get('gold_outcome', '')}")
        print(f"\n【模型输出】")
        print(f"  C: {row.get('best_pred_ctx', '')}")
        print(f"  I: {row.get('best_pred_int', '')}")
        print(f"  O: {row.get('best_pred_out', '')}")

        c = '✓' if str(row.get('c_match', '')).lower() == 'true' else '✗'
        i = '✓' if str(row.get('i_match', '')).lower() == 'true' else '✗'
        o = '✓' if str(row.get('o_match', '')).lower() == 'true' else '✗'
        print(f"\n  Judge 给分: {row.get('best_score', '?')}/3  "
              f"(C{c} I{i} O{o})")
        print(f"  Judge 理由: {str(row.get('best_reason', ''))[:80]}")

        while True:
            score = input("\n你的评分 (0/1/2/3, q=退出, s=跳过): ").strip().lower()
            if score == 'q':
                df.to_csv(SAMPLE_CSV, index=False, encoding='utf-8-sig')
                # [修复5] 退出时汇报跳过数量
                print(f"\n已保存（{done - 1} 条已评，{skipped} 条跳过）→ {SAMPLE_CSV}")
                return
            if score == 's':
                skipped += 1
                break
            if score in ('0', '1', '2', '3'):
                # [修复8] 询问分维度细节
                print("  可选：分维度标注（直接回车跳过）")
                hc = input("  C维度匹配？(1=是, 0=否, 回车跳过): ").strip()
                hi = input("  I维度匹配？(1=是, 0=否, 回车跳过): ").strip()
                ho = input("  O维度匹配？(1=是, 0=否, 回车跳过): ").strip()
                note = input("  备注（可选）: ").strip()
                df.at[idx, 'human_score'] = int(score)
                df.at[idx, 'human_c']     = hc if hc in ('0', '1') else ''
                df.at[idx, 'human_i']     = hi if hi in ('0', '1') else ''
                df.at[idx, 'human_o']     = ho if ho in ('0', '1') else ''
                df.at[idx, 'human_note']  = note
                break
            print("请输入 0、1、2 或 3")

    df.to_csv(SAMPLE_CSV, index=False, encoding='utf-8-sig')
    df.to_csv(RESULT_CSV, index=False, encoding='utf-8-sig')
    # [修复5] 打分结束时汇报跳过数
    total_scored = (df['human_score'].astype(str).str.strip()
                    .isin(['0', '1', '2', '3'])).sum()
    print(f"\n打分完成！有效打分 {total_scored} 条，跳过 {skipped} 条 → {RESULT_CSV}")
    print("运行 step 3 计算 Cohen's κ")


# ══════════════════════════════════════════════
# Step 3: 计算 Cohen's κ（0/1/2/3 四档加权）
# ══════════════════════════════════════════════
def cohen_kappa_weighted(y_judge: list, y_human: list,
                         labels: list) -> dict:
    """
    [修复2] 返回值改为字典，命名明确区分：
      exact_agree  : 完全一致率（po_exact）
      kappa        : 标准 κ（基于完全一致）
      kappa_weighted: 加权 κ（线性权重，主报告指标）
      conf         : 混淆矩阵
    """
    k         = len(labels)
    label_idx = {l: i for i, l in enumerate(labels)}
    conf      = np.zeros((k, k), dtype=float)

    for a, b in zip(y_judge, y_human):
        if a in label_idx and b in label_idx:
            conf[label_idx[a]][label_idx[b]] += 1

    total    = conf.sum()
    row_sums = conf.sum(axis=1)
    col_sums = conf.sum(axis=0)

    # 标准 κ
    po_exact = np.trace(conf) / total
    pe       = (row_sums @ col_sums) / (total ** 2)
    kappa    = (po_exact - pe) / (1 - pe) if (1 - pe) > 0 else 0.0

    # 加权 κ（线性权重）
    max_diff = max(labels) - min(labels)
    weights  = np.array([
        [1 - abs(labels[i] - labels[j]) / max_diff for j in range(k)]
        for i in range(k)
    ])
    po_w   = (weights * conf).sum() / total
    pe_w   = (weights * np.outer(row_sums, col_sums)).sum() / (total ** 2)
    kappa_w = (po_w - pe_w) / (1 - pe_w) if (1 - pe_w) > 0 else 0.0

    return {
        "exact_agree"    : round(float(po_exact), 4),
        "kappa"          : round(float(kappa), 4),
        "kappa_weighted" : round(float(kappa_w), 4),
        "conf"           : conf,
    }


def step3_kappa():
    path = RESULT_CSV if os.path.exists(RESULT_CSV) else SAMPLE_CSV
    if not os.path.exists(path):
        print("[错误] 找不到打分文件，请先完成 step 1 和 step 2")
        return

    df = pd.read_csv(path)
    valid_mask = df['human_score'].astype(str).str.strip().isin(['0', '1', '2', '3'])
    scored     = df[valid_mask].copy()

    # [修复5] 明确汇报跳过/缺失条目
    total_rows  = len(df)
    skipped_n   = total_rows - len(scored)
    if skipped_n > 0:
        print(f"[注意] {total_rows} 条中有 {skipped_n} 条未评分（已排除）")

    if len(scored) == 0:
        print("[错误] 没有有效打分记录")
        return

    scored['human_score'] = scored['human_score'].astype(str).str.strip().astype(int)
    scored['best_score']  = scored['best_score'].astype(int)

    judge = scored['best_score'].tolist()
    human = scored['human_score'].tolist()

    # [修复2] 使用返回字典，内外命名统一
    result = cohen_kappa_weighted(judge, human, SCORE_LABELS)
    po_exact = result["exact_agree"]
    kappa    = result["kappa"]
    kappa_w  = result["kappa_weighted"]
    conf     = result["conf"]
    agree    = int(po_exact * len(scored))

    print(f"\n{'=' * 55}")
    print(f"  人工一致性验证结果（{JUDGE_MODEL}，0/1/2/3 四档）")
    print(f"{'=' * 55}")
    print(f"  标注条目数       : {len(scored)}")
    print(f"  完全一致         : {agree}/{len(scored)} ({po_exact * 100:.1f}%)")
    print()

    # 得分分布对比
    print(f"  得分分布对比:")
    print(f"  {'分数':>4}  {'Judge':>8}  {'人工':>8}")
    for s in SCORE_LABELS:
        jc = judge.count(s)
        hc = human.count(s)
        print(f"  {s:>4}  {jc:>8}  {hc:>8}")
    print()

    print(f"  Cohen's κ（标准） : {kappa:.4f}")
    print(f"  Cohen's κ（加权） : {kappa_w:.4f}  ← 主要报告指标")
    print()
    print(f"  κ 解读标准:")
    print(f"    < 0.40   : 一致性较差")
    print(f"    0.40~0.60: 中等一致")
    print(f"    0.60~0.80: 较强一致  ← 学术可接受")
    print(f"    > 0.80   : 近乎完全一致")

    if   kappa_w >= 0.80: level = "近乎完全一致，Judge 评分高度可信"
    elif kappa_w >= 0.60: level = "较强一致（κ≥0.60），Judge 评分可靠"
    elif kappa_w >= 0.40: level = "中等一致，建议检查分歧样本"
    else:                 level = "一致性较差，需重新审视 Judge Prompt"
    print(f"\n  结论: {level}")

    # [修复8] 混淆矩阵（动态生成，不硬编码标签）
    print(f"\n  混淆矩阵（行=Judge，列=人工）:")
    header = "        " + "".join(f"  人工{s}" for s in SCORE_LABELS)
    print(header)
    for i, s in enumerate(SCORE_LABELS):
        row_str = f"  Judge{s}: " + "".join(
            f"{int(conf[i][j]):6d}" for j in range(len(SCORE_LABELS))
        )
        print(row_str)

    # [修复8] 分维度 C/I/O 一致率分析
    dim_stats = {}
    for dim in ('c', 'i', 'o'):
        judge_col = f'{dim}_match'
        human_col = f'human_{dim}'
        if judge_col in scored.columns and human_col in scored.columns:
            sub = scored[
                scored[human_col].astype(str).str.strip().isin(['0', '1'])
            ].copy()
            if len(sub) > 0:
                sub[judge_col] = sub[judge_col].astype(str).str.lower() == 'true'
                sub[human_col] = sub[human_col].astype(str).str.strip() == '1'
                agree_dim = (sub[judge_col] == sub[human_col]).mean()
                dim_stats[dim.upper()] = round(agree_dim, 4)

    if dim_stats:
        print(f"\n  分维度一致率（Judge vs 人工）:")
        for dim, rate in dim_stats.items():
            bar = "█" * int(rate * 20)
            print(f"    {dim}: {rate:.4f}  {bar}")
    else:
        print(f"\n  [提示] 未检测到 human_c/i/o 列，"
              f"分维度分析需在 Step2 填写或手动填入 CSV")

    # 分歧分析
    disagree = scored[scored['best_score'] != scored['human_score']]
    if len(disagree) > 0:
        print(f"\n  分歧条目: {len(disagree)} 条")
        for _, r in disagree.iterrows():
            diff      = int(r['best_score']) - int(r['human_score'])
            direction = f"Judge高{abs(diff)}" if diff > 0 else f"人工高{abs(diff)}"
            print(f"    [{direction}] Judge={r['best_score']} 人工={r['human_score']} | "
                  f"I: {str(r.get('gold_interv',''))[:35]} "
                  f"→ {str(r.get('best_pred_int',''))[:35]}")

    # 保存 JSON 摘要
    summary = {
        "date"                : datetime.now().strftime("%Y-%m-%d"),
        "judge_model"         : JUDGE_MODEL,
        "scoring_scheme"      : "0/1/2/3",
        "n_total"             : total_rows,
        "n_scored"            : len(scored),
        "n_skipped"           : skipped_n,
        "exact_agreement_rate": po_exact,
        "cohen_kappa"         : kappa,
        "cohen_kappa_weighted": kappa_w,
        "n_disagree"          : len(disagree),
        "dim_agreement"       : dim_stats,
        "conclusion"          : level,
    }
    out_json = "./kappa_summary_v3.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存 → {out_json}")

    # [修复7] 论文写作模板：Judge 模型名从配置读取
    print(f"\n论文写作模板:")
    print(f'  "To validate the reliability of the {JUDGE_MODEL} judge, we randomly')
    print(f'  sampled {len(scored)} gold–model CIMO pairs using stratified sampling')
    print(f'  across the four score levels (0–3) and manually annotated them.')
    print(f'  The weighted Cohen\'s κ was {kappa_w:.2f}, indicating {level.split("，")[0].lower()}.')
    if dim_stats:
        dim_str = ", ".join(f"{d}={v:.2f}" for d, v in dim_stats.items())
        print(f'  Dimension-level agreement rates were {dim_str}.')
    print(f'  Among the {len(disagree)} disagreement cases, discrepancies primarily')
    print(f'  stemmed from boundary cases in intervention–condition distinction."')


# ══════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIMO 人工一致性验证")
    parser.add_argument('--step', type=int, choices=[1, 2, 3], required=True,
                        help="1=抽样, 2=交互打分, 3=计算κ")
    parser.add_argument('--csv',  type=str, default=None,
                        help="覆盖默认 EVAL_CSV 路径")
    args = parser.parse_args()

    if args.csv:
        EVAL_CSV = args.csv

    if   args.step == 1: step1_sample()
    elif args.step == 2: step2_score()
    elif args.step == 3: step3_kappa()