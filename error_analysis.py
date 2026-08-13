"""
Error Analysis 脚本（修复版 v3）
修复清单：
  [1] analyze_by_theme 改用 theme_num 列直接过滤，不再用 Intervention 文本匹配
  [2] analyze_hard_cases 改用复合键 (gold_title, gold_interv)，并在 merge 前去重，消除笛卡尔积
  [3] infer_theme 关键词与 THEMES 字典对齐，按优先级从严到宽排列，消除 T4/T6 'integration' 冲突
  [4] weakest 维度支持并列：返回所有并列最弱维度（如 "Context/Intervention"）
  [5] 文件路径统一为 v3 后缀，与 eval_cimo_v3.py 输出衔接
  [6] generate_summary 改为数据驱动，去除硬编码推断结论
  [7] 缺失文件统一汇总报告，不再静默跳过
"""

import pandas as pd
import numpy as np
import re
import os

# ══════════════════════════════════════════════
# 1. 配置
# ══════════════════════════════════════════════
GOLD_CSV = "./table4_ground_truth_item.csv"

# [修复5] 路径统一为 v3，与 eval_cimo_v3.py 输出衔接
MODELS = {
    "Qwen3-235B": {
        "recall": "./eval_recall_qwen3_v5.csv",
        "prec"  : "./eval_precision_qwen3_v5.csv",
    },
    "DeepSeek-V3.2": {
        "recall": "./eval_recall_deepseek_v5.csv",
        "prec"  : "./eval_precision_deepseek_v5.csv",
    },
    "LLama3.3-70B": {
        "recall": "./eval_recall_llama_v5.csv",
        "prec"  : "./eval_precision_llama_v5.csv",
    },
    "Rule-based": {
        "recall": "./eval_recall_rulebased_v5.csv",
        "prec"  : "./eval_precision_rulebased_v5.csv",
    },
}

BASELINE_MODELS = {"Rule-based", "Random"}   # 不纳入 LLM 均分计算

OUT_THEME   = "./error_analysis_theme_v3.csv"
OUT_HARD    = "./error_analysis_hard_v3.csv"
OUT_FP      = "./error_analysis_fp_v3.csv"
OUT_SUMMARY = "./error_analysis_summary_v3.txt"

THEME_MAP = {
    "1": "T1: Governance mode",
    "2": "T2: Network formation",
    "3": "T3: Interorg. relationships",
    "4": "T4: Strategic exploitation",
    "5": "T5: Open innovation",
    "6": "T6: Operational practices",
}


# ══════════════════════════════════════════════
# 2. 工具函数
# ══════════════════════════════════════════════
def norm(t):
    if not isinstance(t, str): return ''
    return re.sub(r'\s+', ' ', t.upper().strip())


def extract_theme_num(theme_str):
    """从 'Theme 3: Interorganizational...' 提取数字"""
    if not isinstance(theme_str, str): return None
    m = re.search(r'(\d)', theme_str)
    return m.group(1) if m else None


def infer_theme(intervention: str) -> str:
    """
    [修复3] 关键词与 THEMES 字典对齐，按优先级从严到宽排列：
    - 越前面的规则越具体，避免宽泛词语（如 'integration'）被后面规则截获
    - 与 extraction 代码的 THEMES 字典保持一致
    """
    if not isinstance(intervention, str):
        return '3'
    i = intervention.lower()

    # T1：治理模式决策
    if any(k in i for k in [
        'governance', 'outsourc', 'make-or-buy', 'make or buy',
        'hierarchi', 'market governance', 'formal contract', 'relational contract',
    ]):
        return '1'

    # T2：网络形成与关系发起（先于 T3，'alliance formation' 比 'alliance' 更具体）
    if any(k in i for k in [
        'network formation', 'alliance formation', 'partner select',
        'relationship initiat', 'weak tie', 'strong tie', 'bridging tie',
        'building tie', 'building strong', 'building weak', 'building a great',
        'maintaining a rich portfolio',
    ]):
        return '2'

    # T5：开放创新与学习（先于 T4，NPD 相关词更具体）
    if any(k in i for k in [
        'npd', 'new product', 'innovat', 'absorptive', 'technolog',
        'customer involv', 'supplier involv', 'involving customer',
        'involving supplier', 'investing in technolog',
    ]):
        return '5'

    # T4：战略性利用外部资源（先于 T6，检查具体的供应商发展/联盟词汇）
    if any(k in i for k in [
        'supplier development', 'alliance learning', 'alliance capability',
        'partner complementar', 'supply network', 'experiences and learning',
        'investment in alliance', 'co-operative relationship',
        'integration with supplier', 'integration with customer',
    ]):
        return '4'

    # T3：组织间关系（先于 T6，检查关系管理相关词）
    if any(k in i for k in [
        'trust', 'commitment', 'relational norm', 'socialization',
        'dependency', 'noncoercive', 'coercive', 'relational capital',
        'justice polic', 'adaptive behavior', 'collaboration',
        'attraction', 'supplier satisfaction',
    ]):
        return '3'

    # T6：运营实践（最后检查，'integration' 等宽泛词放在这里兜底）
    if any(k in i for k in [
        'edi', 'information shar', 'e-business', 'integration',
        'communication', 'coordination', 'lead-time', 'agility', 'quality',
        'e-reverse', 'information technolog',
    ]):
        return '6'

    return '3'   # 默认归类到最大主题 T3


def weakest_dims(rc: float, ri: float, ro: float) -> str:
    """
    [修复4] 返回所有并列最弱维度（支持 "Context/Intervention" 这样的并列结果）
    """
    vals = {'C': rc, 'I': ri, 'O': ro}
    min_val = min(vals.values())
    tied = [k for k, v in vals.items() if abs(v - min_val) < 1e-9]
    return '/'.join(tied)


def load_gold_with_theme(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding='utf-8')
    df['_n'] = df['Title'].apply(norm)

    theme_col = next(
        (col for col in df.columns if 'theme' in col.lower() or 'research' in col.lower()),
        None
    )
    if theme_col:
        df['theme_num'] = df[theme_col].apply(extract_theme_num)
        print(f"找到 Theme 列: '{theme_col}'")
        print(df['theme_num'].value_counts().sort_index())
    else:
        print(f"[注意] 未找到 Research Theme 列，用 infer_theme 从 Intervention 推断")
        df['theme_num'] = df['Intervention'].apply(infer_theme)
        print(df['theme_num'].value_counts().sort_index())

    return df


# ══════════════════════════════════════════════
# 3. 按 Theme 分析
# ══════════════════════════════════════════════
def analyze_by_theme(gold_df: pd.DataFrame, models_data: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Analysis 1: 按 Research Theme 分组")
    print("=" * 60)

    rows = []
    for theme_num, theme_name in THEME_MAP.items():
        theme_gold = gold_df[gold_df['theme_num'] == theme_num]
        n_gold = len(theme_gold)
        if n_gold == 0:
            continue

        row = {'theme': theme_name, 'n_gold': n_gold}

        for model_name, dfs in models_data.items():
            recall_df = dfs['recall']

            # [修复1] 优先使用 theme_num 列直接过滤（主流程 merge 后已存在）
            if 'theme_num' in recall_df.columns:
                theme_recall = recall_df[recall_df['theme_num'] == theme_num].copy()
            else:
                # 备选：用黄金集的 Intervention 文本过滤（精确匹配，不做模糊）
                print(f"  [警告] {model_name} recall_df 缺少 theme_num 列，改用文本匹配")
                theme_recall = recall_df[
                    recall_df['gold_interv'].isin(theme_gold['Intervention'].tolist())
                ].copy()

            pfx = model_name
            if len(theme_recall) == 0:
                for k in ['recall_strict', 'avg_score', 'recall_C', 'recall_I', 'recall_O']:
                    row[f'{pfx}_{k}'] = None
                continue

            tp3 = (theme_recall['best_score'] == 3).sum()
            row[f'{pfx}_recall_strict'] = round(tp3 / len(theme_recall), 4)
            row[f'{pfx}_avg_score']     = round(theme_recall['best_score'].mean(), 4)

            if 'c_match' in theme_recall.columns:
                row[f'{pfx}_recall_C'] = round(theme_recall['c_match'].mean(), 4)
                row[f'{pfx}_recall_I'] = round(theme_recall['i_match'].mean(), 4)
                row[f'{pfx}_recall_O'] = round(theme_recall['o_match'].mean(), 4)
            else:
                row[f'{pfx}_recall_C'] = row[f'{pfx}_recall_I'] = row[f'{pfx}_recall_O'] = None

        rows.append(row)

        print(f"\n{theme_name} (n={n_gold}):")
        for model_name in models_data:
            s = row.get(f'{model_name}_recall_strict')
            a = row.get(f'{model_name}_avg_score')
            rc = row.get(f'{model_name}_recall_C')
            ri = row.get(f'{model_name}_recall_I')
            ro = row.get(f'{model_name}_recall_O')
            if s is not None:
                cio = (f"  C={rc:.3f} I={ri:.3f} O={ro:.3f}"
                       if rc is not None else "")
                print(f"  {model_name:<18} R@3={s:.4f}  avg={a:.4f}{cio}")

    df_theme = pd.DataFrame(rows)
    df_theme.to_csv(OUT_THEME, index=False, encoding='utf-8-sig')
    print(f"\n→ {OUT_THEME}")
    return df_theme


# ══════════════════════════════════════════════
# 4. 最难匹配条目分析
# ══════════════════════════════════════════════
def analyze_hard_cases(models_data: dict) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Analysis 2: 最难匹配条目（多模型均低分）")
    print("=" * 60)

    llm_models = [m for m in models_data if m not in BASELINE_MODELS]
    if not llm_models:
        print("[跳过] 无 LLM 模型数据")
        return pd.DataFrame()

    # [修复2] 每个模型先按复合键 (gold_title, gold_interv) 去重（取 min score 保守估计）
    #         再依次 merge，消除笛卡尔积
    merged = None
    for m in llm_models:
        df = models_data[m]['recall'].copy()
        # 去重：同一 (gold_title, gold_interv) 只保留最低分（最保守估计）
        dedup = (
            df.groupby(['gold_title', 'gold_interv'], as_index=False)
              .agg(best_score=('best_score', 'min'))
        )
        dedup = dedup.rename(columns={'best_score': f'score_{m}'})

        if merged is None:
            # 保留第一个模型的上下文列
            ctx_cols = ['gold_title', 'gold_context', 'gold_interv',
                        'gold_outcome', 'gold_theme']
            ctx_cols = [c for c in ctx_cols if c in df.columns]
            ctx = (
                df[ctx_cols + ['best_score']]
                  .sort_values('best_score')
                  .drop_duplicates(subset=['gold_title', 'gold_interv'], keep='first')
                  .drop(columns='best_score')
            )
            merged = ctx.merge(dedup, on=['gold_title', 'gold_interv'], how='left')
        else:
            merged = merged.merge(dedup, on=['gold_title', 'gold_interv'], how='left')

    score_cols = [f'score_{m}' for m in llm_models]
    merged['avg_llm_score'] = merged[score_cols].mean(axis=1)

    # 阈值：所有 LLM 平均得分 ≤ 1
    hard = merged[merged['avg_llm_score'] <= 1.0].sort_values('avg_llm_score')
    print(f"所有 LLM 平均得分 ≤ 1 的条目: {len(hard)} 条")
    for _, r in hard.head(10).iterrows():
        scores_str = "  ".join(f"{m}={r[f'score_{m}']:.0f}" for m in llm_models)
        print(f"  avg={r['avg_llm_score']:.2f} [{scores_str}] | "
              f"I: {str(r['gold_interv'])[:55]}")

    out_cols = (['gold_title', 'gold_context', 'gold_interv', 'gold_outcome']
                + [c for c in ['gold_theme'] if c in merged.columns]
                + ['avg_llm_score'] + score_cols)
    out_cols = [c for c in out_cols if c in hard.columns]
    hard[out_cols].head(20).to_csv(OUT_HARD, index=False, encoding='utf-8-sig')
    print(f"\n→ {OUT_HARD}")
    return hard


# ══════════════════════════════════════════════
# 5. FP 错误类型分析
# ══════════════════════════════════════════════
def analyze_fp(models_data: dict):
    print("\n" + "=" * 60)
    print("Analysis 3: FP 错误分析")
    print("=" * 60)

    bad_starts = frozenset([
        'high', 'low', 'higher', 'lower', 'the', 'a', 'an',
        'increased', 'decreased', 'greater', 'less', 'presence',
        'strong', 'weak', 'large', 'small', 'greater', 'limited',
    ])

    fp_rows = []
    for model_name, dfs in models_data.items():
        prec_df = dfs.get('prec')
        if prec_df is None:
            continue

        fp_df = prec_df[prec_df['label'] == 'FP'].copy()
        fp_df['model'] = model_name
        fp_rows.append(fp_df)

        n_fp  = len(fp_df)
        n_all = len(prec_df[prec_df['label'] != 'ERR'])
        print(f"\n{model_name}: FP={n_fp}条 / 总预测={n_all}条"
              f"  FP率={n_fp/n_all*100:.1f}%" if n_all > 0 else f"\n{model_name}: FP={n_fp}条")

        if n_fp == 0:
            continue

        # 分析1：Intervention 非动词开头
        bad_interv = fp_df['pred_interv'].astype(str).apply(
            lambda x: x.lower().split()[0] in bad_starts if x.split() else False
        ).sum()
        print(f"  Intervention 非动词开头: {bad_interv}条 ({bad_interv/n_fp*100:.1f}%)")

        # 分析2：FP 中 VN 误判（reason 中含特定词）
        if 'reason' in fp_df.columns:
            unsupported = fp_df['reason'].astype(str).str.lower().str.contains(
                'not support|not in paper|contradict', na=False
            ).sum()
            print(f"  论文不支持的 FP: {unsupported}条 ({unsupported/n_fp*100:.1f}%)")

        # 显示前5条
        for _, r in fp_df.head(5).iterrows():
            print(f"  FP | I: {str(r.get('pred_interv',''))[:60]}")
            print(f"       R: {str(r.get('reason',''))[:60]}")

    if fp_rows:
        pd.concat(fp_rows).to_csv(OUT_FP, index=False, encoding='utf-8-sig')
        print(f"\n→ {OUT_FP}")


# ══════════════════════════════════════════════
# 6. 生成文字摘要（数据驱动）
# ══════════════════════════════════════════════
def generate_summary(df_theme: pd.DataFrame, models_data: dict):
    lines = []
    lines.append("Error Analysis Summary")
    lines.append("=" * 60)
    lines.append("")

    llm_models = [m for m in models_data if m not in BASELINE_MODELS]

    # ── 主题难度排序 ──────────────────────────────
    if llm_models and df_theme is not None and len(df_theme) > 0:
        score_cols = [f'{m}_avg_score' for m in llm_models
                      if f'{m}_avg_score' in df_theme.columns]
        if score_cols:
            df_theme = df_theme.copy()
            df_theme['mean_llm_score'] = df_theme[score_cols].mean(axis=1)
            df_sorted = (df_theme.dropna(subset=['mean_llm_score'])
                                 .sort_values('mean_llm_score'))

            lines.append("【主题难度排序（从难到易）】")
            for _, r in df_sorted.iterrows():
                lines.append(
                    f"  {r['theme']:<35} "
                    f"avg_score={r['mean_llm_score']:.4f}/3.0  "
                    f"n={int(r['n_gold'])}"
                )

            if len(df_sorted) >= 2:
                hardest = df_sorted.iloc[0]
                easiest = df_sorted.iloc[-1]
                score_gap = easiest['mean_llm_score'] - hardest['mean_llm_score']
                lines.append("")
                lines.append(f"最难主题: {hardest['theme']} "
                              f"(avg={hardest['mean_llm_score']:.4f})")
                lines.append(f"最易主题: {easiest['theme']} "
                              f"(avg={easiest['mean_llm_score']:.4f})")
                lines.append(f"难易差值: {score_gap:.4f} / 3.0")

                # [修复6] 数据驱动的论文模板，不出现硬编码推断
                lines.append("")
                lines.append("【论文 Discussion 素材（数据驱动）】")
                lines.append(
                    f"Theme-level analysis shows that {hardest['theme']} "
                    f"was the most difficult theme for all LLMs "
                    f"(mean score = {hardest['mean_llm_score']:.2f}/3.0, "
                    f"n = {int(hardest['n_gold'])}), "
                    f"while {easiest['theme']} achieved the highest scores "
                    f"(mean score = {easiest['mean_llm_score']:.2f}/3.0, "
                    f"n = {int(easiest['n_gold'])}). "
                    f"The gap between the hardest and easiest theme was "
                    f"{score_gap:.2f} points on the 0–3 scale."
                )

    # ── C/I/O 维度难度 ────────────────────────────
    lines.append("")
    lines.append("【C/I/O 维度难度（各模型）】")
    for model_name, dfs in models_data.items():
        if model_name in BASELINE_MODELS:
            continue
        r_df = dfs['recall']
        if 'c_match' not in r_df.columns:
            continue
        rc = r_df['c_match'].mean()
        ri = r_df['i_match'].mean()
        ro = r_df['o_match'].mean()
        # [修复4] 支持并列最弱维度
        wd = weakest_dims(rc, ri, ro)
        lines.append(
            f"  {model_name}: C={rc:.4f} I={ri:.4f} O={ro:.4f}  "
            f"最弱维度={wd}"
        )

    # ── 各模型 Theme 分维度热力表 ─────────────────
    lines.append("")
    lines.append("【各主题 × C/I/O 平均匹配率（LLM 均值）】")
    if df_theme is not None:
        for _, r in df_theme.iterrows():
            theme = r['theme']
            for dim in ('C', 'I', 'O'):
                vals = [r.get(f'{m}_recall_{dim}') for m in llm_models
                        if r.get(f'{m}_recall_{dim}') is not None]
                avg  = np.mean(vals) if vals else float('nan')
                lines.append(f"  {theme:<35} {dim}: {avg:.4f}")

    with open(OUT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n→ {OUT_SUMMARY}")


# ══════════════════════════════════════════════
# 7. 主流程
# ══════════════════════════════════════════════
def run():
    print("加载黄金集...")
    gold_df = load_gold_with_theme(GOLD_CSV)

    # 构建 Intervention → theme_num 映射（去重，用于 merge）
    gold_interv_theme = (
        gold_df[['Intervention', 'theme_num']]
        .drop_duplicates(subset=['Intervention'])
    )

    print("\n加载评测结果...")
    models_data  = {}
    missing_files = []   # [修复7] 统一收集缺失文件

    for model_name, paths in MODELS.items():
        recall_path = paths['recall']
        prec_path   = paths['prec']

        if not os.path.exists(recall_path):
            missing_files.append(f"  [{model_name}] recall: {recall_path}")
            continue

        recall_df = pd.read_csv(recall_path)

        # [修复1] merge theme_num 到 recall_df
        # 若 recall_df 已有 gold_theme 列（来自 eval_cimo_v3.py），优先用它
        if 'gold_theme' in recall_df.columns and 'theme_num' not in recall_df.columns:
            recall_df['theme_num'] = recall_df['gold_theme'].apply(extract_theme_num)
        elif 'theme_num' not in recall_df.columns:
            recall_df = recall_df.merge(
                gold_interv_theme,
                left_on='gold_interv', right_on='Intervention',
                how='left'
            )

        prec_df = None
        if not os.path.exists(prec_path):
            missing_files.append(f"  [{model_name}] prec : {prec_path}")
        else:
            prec_df = pd.read_csv(prec_path)

        models_data[model_name] = {'recall': recall_df, 'prec': prec_df}
        avg = recall_df['best_score'].mean() if 'best_score' in recall_df.columns else 0
        theme_coverage = recall_df['theme_num'].notna().mean() if 'theme_num' in recall_df.columns else 0
        print(f"  {model_name}: {len(recall_df)}条  "
              f"avg_score={avg:.4f}  theme覆盖率={theme_coverage:.1%}")

    # [修复7] 统一打印缺失文件列表
    if missing_files:
        print(f"\n[缺失文件，已跳过]")
        for f in missing_files:
            print(f)

    if not models_data:
        print("[错误] 没有找到任何评测结果文件，请检查路径配置")
        return

    # 运行分析
    df_theme = analyze_by_theme(gold_df, models_data)
    analyze_hard_cases(models_data)
    analyze_fp(models_data)
    generate_summary(df_theme, models_data)

    print("\n" + "=" * 60)
    print("Error Analysis 完成！")
    print(f"  主题分析  → {OUT_THEME}")
    print(f"  难例分析  → {OUT_HARD}")
    print(f"  FP 分析   → {OUT_FP}")
    print(f"  文字摘要  → {OUT_SUMMARY}")
    print("=" * 60)


if __name__ == "__main__":
    run()