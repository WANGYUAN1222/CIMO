import os
# 【注意：必须在 import torch 之前设置显卡环境变量】
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2" # 强制只让代码看到两张卡

import json
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# ======= 引入原生的 vLLM =======
from vllm import LLM, SamplingParams

# ================= 配置区域 =================
MODEL_ID_OR_PATH = "/data_share_from_3090/wy_code/code/EE/NER/utd24/LLM-Research/Llama-3.3-70B-Instruct" 
PATH_NEG = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/0"  
PATH_POS = "/data_share_from_3090/wy_code/code/EE/NER/utd24/output/1"  
OUTPUT_FILE = "paper_md_inference_result.csv"
BATCH_SIZE = 1  # 🌟 新增：每个批次处理的论文数量，可根据实际情况调整

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

# ================= 数据读取与解析 =================
def load_markdown_files(root_dir, label):
    data = []
    if not os.path.exists(root_dir):
        return data
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".md"):
                filepath = os.path.join(dirpath, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content:
                        data.append({
                            "filename": filename,
                            "filepath": filepath,
                            "content": content,
                            "true_label": label
                        })
                except Exception as e:
                    pass
    return data

def parse_json_response(response_text):
    try:
        return json.loads(response_text)
    except:
        try:
            import re
            match = re.search(r"```json(.*?)```", response_text, re.DOTALL)
            if match: return json.loads(match.group(1).strip())
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end != -1: return json.loads(response_text[start:end])
        except:
            pass
    return None

# ================= 主程序 =================
def main():
    print("=== 开始加载数据 ===")
    all_data = load_markdown_files(PATH_NEG, 0) + load_markdown_files(PATH_POS, 1)
    if len(all_data) == 0: 
        print("未找到任何 markdown 文件，请检查路径。")
        return

    num_gpus = torch.cuda.device_count()
    print(f"\n=== 检测到 {num_gpus} 张可用显卡，正在启动原生 vLLM 引擎 ===")
    
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
    
    print(f"模型加载完成。共有 {len(all_data)} 条数据待处理。")
    print(f"采用分批循环处理，每批次大小: {BATCH_SIZE}...")

    # 🌟 初始化输出文件，写入表头（覆盖旧文件）
    columns = ["filename", "true_label", "pred_label", "reasoning", "raw_response"]
    pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False, encoding='utf_8_sig')

    all_results = []

    # 🌟 循环分批处理
    for i in tqdm(range(0, len(all_data), BATCH_SIZE), desc="整体批次进度"):
        batch_data = all_data[i : i + BATCH_SIZE]
        
        # 1. 构建当前批次的对话
        conversations = []
        for item in batch_data:
            content_preview = item['content'][:MAX_INPUT_CHARS]
            input_text = f"论文内容片段：\n{content_preview}..."
            
            conversations.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": input_text}
            ])
        
        # 2. 执行当前批次推理
        # 注意：vLLM 内部会有自己的小进度条
        responses = llm.chat(messages=conversations, sampling_params=sampling_params)
        
        # 3. 解析当前批次结果
        batch_results = []
        for item, resp in zip(batch_data, responses):
            response_text = resp.outputs[0].text
            parsed = parse_json_response(response_text)
            
            pred_label = -1
            reasoning = "Parse Error"
            
            if parsed and "decision" in parsed:
                pred_label = int(parsed["decision"])
                reasoning = parsed.get("reasoning", "")
            else:
                if "1" in response_text and "Include" in response_text: pred_label = 1
                elif "0" in response_text and "Exclude" in response_text: pred_label = 0
                reasoning = response_text  

            result_dict = {
                "filename": item['filename'],
                "true_label": item['true_label'],
                "pred_label": pred_label,
                "reasoning": reasoning,
                "raw_response": response_text 
            }
            batch_results.append(result_dict)
            all_results.append(result_dict) # 汇总用于最后算准确率
            
        # 4. 🌟 边推边存：将当前批次结果追加写入 CSV 
        # mode='a' 表示 append，header=False 表示不再重复写入表头
        pd.DataFrame(batch_results).to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf_8_sig')

    print(f"\n✅ 所有数据推理完毕！完整结果已保存至: {OUTPUT_FILE}")

    # ================= 最终评估 =================
    df = pd.DataFrame(all_results)
    valid_df = df[df['pred_label'] != -1]
    
    if not valid_df.empty:
        acc = accuracy_score(valid_df['true_label'], valid_df['pred_label'])
        print(f"\n准确率 (Accuracy): {acc:.2%}")
        print(classification_report(valid_df['true_label'], valid_df['pred_label'], target_names=['Exclude(0)', 'Include(1)']))

if __name__ == "__main__":
    main()