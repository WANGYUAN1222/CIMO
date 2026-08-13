import os
# 【注意：必须在 import torch 之前设置显卡环境变量】
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"  # 强制只让代码看到两张卡

import json
import shutil
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# ======= 引入原生的 vLLM =======
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
MODEL_ID_OR_PATH = "/data_share_from_3090/wy_code/code/EE/NER/utd24/LLM-Research/Llama-3.3-70B-Instruct"

# 📁 新数据源：EBM 文件夹（按期刊/年份/论文UUID组织）
PATH_EBM = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/EBM"

# 📁 分类输出目录（预测为0/1的论文将被复制到对应子目录）
OUTPUT_DIR_NEG = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/classified/0"
OUTPUT_DIR_POS = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/classified/1"

OUTPUT_FILE = "paper_md_inference_result.csv"
BATCH_SIZE = 1  # 每批次处理的论文数量

# 控制是否复制整个论文文件夹（True）或仅复制 .md 文件（False）
COPY_ENTIRE_FOLDER = True

SYSTEM_PROMPT = """
You are a professional management researcher. Please determine whether the paper should be included in the "External Resource Management" literature review based on the provided paper content (typically containing title, abstract, and introduction).

[Inclusion Criteria]
1. **Topic**: The research focus must be on "inter-organizational relations," such as buyer-supplier relationships, strategic alliances, supply chain networks, etc.
   - *Exclude*: Purely internal corporate management (e.g., internal HR, production scheduling) or B2C market research (consumer behavior).
2. **Theory**: The paper must explicitly involve theory development or theory testing.
   - *Exclude*: Purely descriptive practical reports or editorials.
3. **Perspective**: Must adopt a "managerial viewpoint," i.e., from the perspective of business managers.

Please strictly output your decision in JSON format without any pleasantries:
{
    "reasoning": "One-sentence reason",
    "decision": 1 
}
(Note: 1 indicates Include, 0 indicates Exclude)
"""

# ================= 数据读取：EBM 目录结构 =================
# 目录结构：EBM / Journal / Year / PaperUUID / hybrid_auto / PaperUUID.md
def load_ebm_markdown_files(ebm_root):
    """
    遍历 EBM 目录，找到每篇论文对应的 .md 文件。
    返回包含文件路径、期刊、年份等元信息的列表。
    """
    data = []
    if not os.path.exists(ebm_root):
        print(f"[ERROR] EBM 路径不存在: {ebm_root}")
        return data

    for journal in sorted(os.listdir(ebm_root)):
        journal_path = os.path.join(ebm_root, journal)
        if not os.path.isdir(journal_path):
            continue

        for year in sorted(os.listdir(journal_path)):
            year_path = os.path.join(journal_path, year)
            if not os.path.isdir(year_path):
                continue

            for paper_uuid in sorted(os.listdir(year_path)):
                paper_path = os.path.join(year_path, paper_uuid)
                if not os.path.isdir(paper_path):
                    continue

                # 在 paper_path 及其子目录（如 hybrid_auto）中寻找 .md 文件
                md_file = None
                md_path = None

                # 优先在 hybrid_auto 子目录中查找
                hybrid_auto_path = os.path.join(paper_path, "hybrid_auto")
                if os.path.isdir(hybrid_auto_path):
                    for fname in os.listdir(hybrid_auto_path):
                        if fname.endswith(".md"):
                            md_file = fname
                            md_path = os.path.join(hybrid_auto_path, fname)
                            break

                # 若 hybrid_auto 中未找到，在 paper_path 根目录查找
                if md_path is None:
                    for fname in os.listdir(paper_path):
                        if fname.endswith(".md"):
                            md_file = fname
                            md_path = os.path.join(paper_path, fname)
                            break

                if md_path is None:
                    continue  # 该论文没有 .md 文件，跳过

                try:
                    with open(md_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if not content:
                        continue

                    data.append({
                        "filename":     md_file,
                        "filepath":     md_path,
                        "paper_uuid":   paper_uuid,
                        "paper_dir":    paper_path,   # 整个论文文件夹路径
                        "journal":      journal,
                        "year":         year,
                        "content":      content,
                        "true_label":   -1            # EBM 数据尚无真实标签
                    })
                except Exception as e:
                    print(f"[WARN] 读取失败: {md_path} | 原因: {e}")

    return data


# ================= 工具函数 =================
def parse_json_response(response_text):
    try:
        return json.loads(response_text)
    except:
        try:
            import re
            match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
            if match:
                return json.loads(match.group(1).strip())
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != -1:
                return json.loads(response_text[start:end])
        except:
            pass
    return None


def save_classified_paper(item, pred_label, journal, year, paper_uuid):
    """
    将论文按预测结果复制到 classified/0 或 classified/1 目录下，
    保留 Journal / Year / PaperUUID 的三级结构。
    """
    base_dir = OUTPUT_DIR_POS if pred_label == 1 else OUTPUT_DIR_NEG

    # 目标路径：classified/{0|1}/Journal/Year/PaperUUID
    dest_paper_dir = os.path.join(base_dir, journal, year, paper_uuid)

    try:
        if COPY_ENTIRE_FOLDER:
            # 复制整个论文文件夹
            if os.path.exists(dest_paper_dir):
                shutil.rmtree(dest_paper_dir)
            shutil.copytree(item["paper_dir"], dest_paper_dir)
        else:
            # 仅复制 .md 文件
            os.makedirs(dest_paper_dir, exist_ok=True)
            dest_md = os.path.join(dest_paper_dir, item["filename"])
            shutil.copy2(item["filepath"], dest_md)

    except Exception as e:
        print(f"[WARN] 复制失败: {item['paper_dir']} → {dest_paper_dir} | 原因: {e}")


# ================= 主程序 =================
def main():
    # ── 创建输出目录 ──────────────────────────────────────────
    os.makedirs(OUTPUT_DIR_NEG, exist_ok=True)
    os.makedirs(OUTPUT_DIR_POS, exist_ok=True)

    print("=== 开始从 EBM 目录加载数据 ===")
    all_data = load_ebm_markdown_files(PATH_EBM)

    if len(all_data) == 0:
        print("未找到任何 markdown 文件，请检查 PATH_EBM 路径。")
        return

    # 打印数据统计
    journals = {}
    for item in all_data:
        journals[item["journal"]] = journals.get(item["journal"], 0) + 1
    print(f"共加载 {len(all_data)} 篇论文，来自 {len(journals)} 个期刊：")
    for j, cnt in sorted(journals.items()):
        print(f"  {j}: {cnt} 篇")

    # ── 加载模型 ─────────────────────────────────────────────
    num_gpus = torch.cuda.device_count()
    print(f"\n=== 检测到 {num_gpus} 张可用显卡，正在启动 vLLM 引擎 ===")

    MAX_INPUT_CHARS = 3000

    llm = LLM(
        model=MODEL_ID_OR_PATH,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=0.95,
        max_model_len=4096,
        trust_remote_code=True,
        enforce_eager=True
    )

    sampling_params = SamplingParams(
        max_tokens=512,
        temperature=0.0
    )

    print(f"模型加载完成。共 {len(all_data)} 条数据，每批 {BATCH_SIZE} 条。\n")

    # ── 初始化 CSV（写入表头）────────────────────────────────
    columns = ["filename", "journal", "year", "paper_uuid",
               "pred_label", "reasoning", "raw_response"]
    pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')

    all_results = []
    stats = {"0": 0, "1": 0, "error": 0}

    # ── 分批推理 ─────────────────────────────────────────────
    for i in tqdm(range(0, len(all_data), BATCH_SIZE), desc="整体批次进度"):
        batch_data = all_data[i: i + BATCH_SIZE]

        # 1. 构建对话
        conversations = []
        for item in batch_data:
            content_preview = item['content'][:MAX_INPUT_CHARS]
            input_text = f"论文内容片段：\n{content_preview}..."
            conversations.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": input_text}
            ])

        # 2. 推理
        responses = llm.chat(messages=conversations, sampling_params=sampling_params)

        # 3. 解析 + 分类保存
        batch_results = []
        for item, resp in zip(batch_data, responses):
            response_text = resp.outputs[0].text
            parsed = parse_json_response(response_text)

            pred_label = -1
            reasoning  = "Parse Error"

            if parsed and "decision" in parsed:
                pred_label = int(parsed["decision"])
                reasoning  = parsed.get("reasoning", "")
            else:
                # 降级：关键词匹配
                if "1" in response_text and "Include" in response_text:
                    pred_label = 1
                elif "0" in response_text and "Exclude" in response_text:
                    pred_label = 0
                reasoning = response_text

            # 统计
            if pred_label == 0:
                stats["0"] += 1
            elif pred_label == 1:
                stats["1"] += 1
            else:
                stats["error"] += 1

            # 📂 按预测结果复制论文到对应文件夹
            if pred_label in (0, 1):
                save_classified_paper(
                    item, pred_label,
                    item["journal"], item["year"], item["paper_uuid"]
                )

            result_dict = {
                "filename":     item["filename"],
                "journal":      item["journal"],
                "year":         item["year"],
                "paper_uuid":   item["paper_uuid"],
                "pred_label":   pred_label,
                "reasoning":    reasoning,
                "raw_response": response_text
            }
            batch_results.append(result_dict)
            all_results.append(result_dict)

        # 4. 追加写入 CSV
        pd.DataFrame(batch_results).to_csv(
            OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf_8_sig'
        )

    # ── 最终汇报 ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"✅ 推理完毕！结果已保存至: {OUTPUT_FILE}")
    print(f"\n📊 分类统计：")
    print(f"  ✅ Include (1): {stats['1']} 篇  →  {OUTPUT_DIR_POS}")
    print(f"  ❌ Exclude (0): {stats['0']} 篇  →  {OUTPUT_DIR_NEG}")
    print(f"  ⚠️  解析失败:    {stats['error']} 篇")
    print(f"{'='*50}")

    # ── 按期刊维度统计 ────────────────────────────────────────
    df = pd.DataFrame(all_results)
    print("\n📋 各期刊分类结果：")
    summary = (
        df[df["pred_label"].isin([0, 1])]
        .groupby(["journal", "pred_label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "Exclude(0)", 1: "Include(1)"})
    )
    summary["Total"] = summary.sum(axis=1)
    summary["Include率"] = (summary.get("Include(1)", 0) / summary["Total"]).map("{:.1%}".format)
    print(summary.to_string())

    # 保存按期刊汇总的统计表
    summary_file = "classification_summary_by_journal.csv"
    summary.to_csv(summary_file, encoding='utf_8_sig')
    print(f"\n📁 期刊汇总表已保存至: {summary_file}")


if __name__ == "__main__":
    main()