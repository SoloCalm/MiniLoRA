"""
模块 5：推理和 base vs LoRA 对比

目标：加载 base 模型和 LoRA adapter，对比生成效果。

运行方式：
    python scripts/my_infer_compare.py --question "高血压患者日常生活中应该注意什么？"

参考代码：scripts/infer_compare.py（先看懂，关掉，再自己写）
"""

import argparse
import gc
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"


def load_model(model_name, adapter_dir=None):
    """加载模型，支持 base 和 LoRA 两种模式

    参数：
        model_name: 模型路径，如 "models/Qwen2.5-0.5B-Instruct"
        adapter_dir: LoRA adapter 路径，None 表示加载 base 模型

    返回：(tokenizer, model)

    步骤：
    1. 从 adapter_dir（如有）或 model_name 加载 tokenizer
    2. 设置 pad_token = eos_token（如果没有的话）
    3. 用 AutoModelForCausalLM 加载模型（bf16，device_map="auto"）
    4. 如果 adapter_dir 不为 None，用 PeftModel.from_pretrained 加载 LoRA
    5. model.eval() 切到推理模式
    """
    # TODO: 你的代码
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Qwen 没有 pad_token，用 eos 代替

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return tokenizer, model

    pass


def generate(tokenizer, model, question, max_new_tokens=256):
    """用模型生成回答

    参数：
        tokenizer: 分词器
        model: 模型
        question: 用户问题
        max_new_tokens: 最大生成 token 数

    返回：str（生成的回答文本）

    步骤：
    1. 构造 messages（system + user，不包含 assistant）
    2. 用 apply_chat_template tokenize（return_tensors="pt"）
    3. 注意 apply_chat_template 可能返回 BatchEncoding，需要检查 .input_ids
    4. inputs 移到 model.device
    5. 用 torch.no_grad() + model.generate() 生成
       参数建议：do_sample=True, temperature=0.7, top_p=0.9
    6. 只取新生成的 token（去掉 prompt 部分），解码返回
    """
    # TODO: 你的代码
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"请以谨慎、专业、易懂的方式回答下面的医疗健康问题。\n\n问题：{question}"},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
    pass


def main():
    parser = argparse.ArgumentParser(description="Compare base and LoRA responses.")
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--question", default="高血压患者日常生活中应该注意什么？")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    # TODO: 步骤 1 - 加载 base 模型，生成回答
    # TODO: 步骤 2 - del model + gc.collect() + torch.cuda.empty_cache() 释放显存
    #         6GB 显存放不下两个模型，必须先释放再加载
    # TODO: 步骤 3 - 检查 adapter_dir 是否存在，加载 LoRA 模型，生成回答

    print("=" * 60)
    print("Base vs LoRA 推理对比")
    print("=" * 60)
    print(f"\n问题: {args.question}\n")

    # 步骤 1 - 加载 base 模型，生成回答
    print("[1/2] 加载 Base 模型...")
    tokenizer_base, model_base = load_model(args.model_name, adapter_dir=None)
    print("  Base 模型加载完成\n")

    print("  Base 模型回答:")
    base_response = generate(
        tokenizer_base,
        model_base,
        args.question,
        args.max_new_tokens,
    )
    print(f"  {base_response}\n")

    # 步骤 2 - 释放显存
    del model_base
    del tokenizer_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 步骤 3 - 检查 adapter_dir 是否存在，加载 LoRA 模型，生成回答
    print("[2/2] 加载 LoRA 模型...")
    if not args.adapter_dir.exists():
        print(f"  LoRA adapter 不存在: {args.adapter_dir}")
        print("  请先运行训练脚本 my_train_lora.py 生成 LoRA adapter")
        return

    tokenizer_lora, model_lora = load_model(args.model_name, args.adapter_dir)
    print("  LoRA 模型加载完成\n")

    print("  LoRA 模型回答:")
    lora_response = generate(
        tokenizer_lora,
        model_lora,
        args.question,
        args.max_new_tokens,
    )
    print(f"  {lora_response}\n")

    print("=" * 60)
    print("对比总结")
    print("=" * 60)
    print(f"\nBase 模型输出长度: {len(base_response)} 字符")
    print(f"LoRA 模型输出长度: {len(lora_response)} 字符")
    pass


if __name__ == "__main__":
    main()
