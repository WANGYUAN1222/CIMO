import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DS_BUILD_OPS"] = "0"

# ✅ 正确的 patch 方式，包装 AdamW 兼容 adamw_mode 参数
from torch.optim import AdamW as TorchAdamW

class CompatCPUAdam(TorchAdamW):
    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 amsgrad=False, adamw_mode=True, **kwargs):
        # ✅ 过滤掉 DeepSpeed 专属参数，只传 torch AdamW 支持的参数
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )

import deepspeed.ops.adam as ds_adam
ds_adam.DeepSpeedCPUAdam = CompatCPUAdam
ds_adam.FusedAdam = CompatCPUAdam

# 后续代码不变...
import sys, types
try:
    import triton.ops
except (ImportError, ModuleNotFoundError):
    import triton
    if not hasattr(triton, "ops"):
        triton.ops = types.ModuleType("triton.ops")
        sys.modules["triton.ops"] = triton.ops

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
from modelscope import snapshot_download
model_dir = snapshot_download("Qwen/Qwen3.5-9B", cache_dir="./")

MODEL_ID = "./Qwen/Qwen3___5-9B"
TRAIN_DATA = "train.jsonl"
OUTPUT_DIR = "./qwen3.5-9b-cimo"    
MAX_SEQ_LENGTH = 30000 

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files={"train": TRAIN_DATA})["train"]

    def formatting_prompts_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        deepspeed="ds_config_zero3.json",
        save_steps=20,
        save_total_limit=3,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        report_to="none",
        optim="adamw_torch",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

if __name__ == "__main__":
    main()