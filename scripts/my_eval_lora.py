"""
模块 6：批量评测 eval_lora.py

目标：批量跑 base vs LoRA 对比，保存结果到 results/base_vs_lora.jsonl

运行方式：
    python scripts/my_eval_lora.py --model-name models/Qwen2.5-0.5B-Instruct

本模块没有参考代码，需要你自己写。
可以复用模块 5 的 load_model 和 generate 函数。
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"


def load_model(model_name, adapter_dir=None):
    """加载模型（复用模块 5）

    adapter_dir=None  → 加载 base 模型
    adapter_dir="路径" → 加载 base + LoRA adapter
    """
    # TODO: 你的代码（从模块 5 复制过来）
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
    """用模型生成回答（复用模块 5）"""
    # TODO: 你的代码（从模块 5 复制过来）
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


def load_jsonl(path):
    """读取 jsonl 文件，返回 list[dict]"""
    # TODO: 你的代码
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()           # 去掉每行末尾的换行符和空白
            if not line:                   # 跳过空行
                continue
            try:
                items.append(json.loads(line))  # 把 JSON 字符串解析成 Python 字典
            except json.JSONDecodeError:
                continue                   # 如果某行 JSON 格式有误，跳过而不是报错
    return items
    pass


def write_jsonl(path, items):
    """写出 jsonl 文件，ensure_ascii=False 保证中文不被转义"""
    # TODO: 你的代码
    path.parent.mkdir(parents=True, exist_ok=True)  # 如果目录不存在就创建
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            # json.dumps 把字典转成 JSON 字符串
            # ensure_ascii=False 保证中文正常显示，不会变成 中文
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    pass
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/medical/eval_prompts.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("results/base_vs_lora.jsonl"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.eval_file.exists():
        print(f"评测文件不存在: {args.eval_file}")
        print("请先创建 data/medical/eval_prompts.jsonl")
        print("每行一个 JSON: {\"question\": \"你的问题\"}")
        return

    # TODO: 步骤 1 - 读取评测问题
    # TODO: 步骤 2 - 加载 base 模型，对所有问题生成回答
    #         用 list 存结果: [{"question": ..., "base_answer": ...}, ...]
    # TODO: 步骤 3 - 释放显存（del + gc.collect + empty_cache）
    # TODO: 步骤 4 - 加载 LoRA 模型，对所有问题生成回答
    #         追加到已有结果: result["lora_answer"] = ...
    # TODO: 步骤 5 - 用 write_jsonl 保存结果到 jsonl
    # 步骤 1 - 读取评测问题
    eval_items = load_jsonl(args.eval_file)
    questions = [item["question"] for item in eval_items if "question" in item]

    if not questions:
        print(f"评测文件中没有有效问题: {args.eval_file}")
        return

    print(f"读取到 {len(questions)} 个评测问题")

    # 步骤 2 - 加载 base 模型，对所有问题生成回答
    print("\n[1/2] 加载 Base 模型...")
    tokenizer_base, model_base = load_model(args.model_name, adapter_dir=None)
    print("  Base 模型加载完成\n")

    results = []
    for i, question in enumerate(questions, start=1):
        print(f"  Base 生成中: {i}/{len(questions)}")
        base_answer = generate(
            tokenizer_base,
            model_base,
            question,
            args.max_new_tokens,
        )
        results.append({
            "question": question,
            "base_answer": base_answer,
        })

    # 步骤 3 - 释放显存
    del model_base
    del tokenizer_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 步骤 4 - 加载 LoRA 模型，对所有问题生成回答
    if not args.adapter_dir.exists():
        print(f"\nLoRA adapter 不存在: {args.adapter_dir}")
        print("请先运行训练脚本生成 LoRA adapter")
        return

    print("\n[2/2] 加载 LoRA 模型...")
    tokenizer_lora, model_lora = load_model(args.model_name, args.adapter_dir)
    print("  LoRA 模型加载完成\n")

    for i, result in enumerate(results, start=1):
        print(f"  LoRA 生成中: {i}/{len(results)}")
        lora_answer = generate(
            tokenizer_lora,
            model_lora,
            result["question"],
            args.max_new_tokens,
        )
        result["lora_answer"] = lora_answer

    # 步骤 5 - 用 write_jsonl 保存结果到 jsonl
    write_jsonl(args.output_file, results)
    print(f"\n评测结果已保存到: {args.output_file}")
    pass


if __name__ == "__main__":
    main()
