import os
import sys
import types
import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ================= 配置 =================
BASE_MODEL_ID = "./LLM-Research/Meta-Llama-3___1-8B-Instruct"
LORA_PATH = "./llama3.1-8b-cimo/checkpoint-9"
TEST_DATA = "test.jsonl"
OUTPUT_FILE = "test_result.jsonl"
MAX_NEW_TOKENS = 2048

def main():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载基座模型...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"加载 LoRA 权重：{LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model.eval()
    print("✅ 模型加载完成")

    # 加载测试数据
    dataset = load_dataset("json", data_files={"test": TEST_DATA})["test"]
    print(f"✅ 测试数据：{len(dataset)} 条")

    results = []
    with torch.no_grad():
        for i, example in enumerate(tqdm(dataset, desc="推理中")):
            # 只取 user 部分作为输入，不包含 assistant 回答
            messages = example["messages"]
            input_messages = [m for m in messages if m["role"] != "assistant"]

            # 构造输入
            input_text = tokenizer.apply_chat_template(
                input_messages,
                tokenize=False,
                add_generation_prompt=True  # ✅ 推理时加上生成提示
            )

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            ).to(model.device)

            # 生成
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,        # ✅ 贪婪解码，结果稳定
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # 只取新生成的部分
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # 获取标准答案
            gold = ""
            for m in messages:
                if m["role"] == "assistant":
                    gold = m["content"]
                    break

            result = {
                "id": i,
                "input": input_messages,
                "prediction": prediction,
                "gold": gold,
            }
            results.append(result)

            # 实时打印前 3 条看效果
            
            print(f"\n--- 样本 {i} ---")
            print(f"预测：{prediction}")
            print(f"标准：{gold}")

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 推理完成，结果保存至：{OUTPUT_FILE}")

if __name__ == "__main__":
    main()