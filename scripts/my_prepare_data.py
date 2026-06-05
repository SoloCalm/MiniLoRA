"""
模块 1：数据准备和预处理 (追加优化版)

目标：从原始 json 文件生成训练用的 jsonl 文件，并提取无 output 的样本追加到评估集中。
优化点：
1. 合并清洗逻辑，单次遍历完成数据分类。
2. 支持 eval_prompts.jsonl 的追加写入，且自动处理换行符防粘连。
"""

import argparse
import json
from pathlib import Path
import random


def read_jsonl(path):
    """读取 jsonl 文件，返回 list[dict]"""
    items: list[dict] = []
    if not path.exists():
        return items
        
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def process_item(item):
    """
    单次遍历处理一条数据：提取公共问题，并根据有无 output 进行分类。
    返回: (sft_item, eval_item)
    """
    instruction = str(item.get("instruction", "")).strip()
    input_text = str(item.get("input", "")).strip()
    output = str(item.get("output", "")).strip()

    if input_text:
        prompt = f"{instruction}\n{input_text}" if instruction else input_text
    else:
        prompt = instruction

    if not prompt:
        return None, None

    if output:
        sft_item = {
            "instruction": "请以谨慎、专业、易懂的方式回答下面的医疗健康问题。",
            "input": prompt,
            "output": output,
        }
        return sft_item, None
    else:
        eval_item = {"question": prompt}
        return None, eval_item


def split_data(items, train_ratio=0.8, valid_ratio=0.1, seed=42):
    """将数据划分为 train / valid / test (本脚本当前直接使用原始分割，此函数备用)"""
    random.seed(seed)
    data = list(items)
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    valid_end = int(n * (train_ratio + valid_ratio))

    return data[:train_end], data[train_end:valid_end], data[valid_end:]


def write_jsonl(path, items, mode="w"):
    """将数据列表写入 jsonl 文件
    
    参数:
        mode: 写入模式。'w' 为覆盖（默认），'a' 为追加。
    """
    # 优化1：如果没有数据，直接返回，避免创建空文件或写入多余空行
    if not items:
        return
        
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 优化2：追加模式下的“防粘连”处理
    # 如果文件已存在且有内容，先补一个换行符，防止新数据的第一行和旧数据的最后一行粘在一起
    if mode == "a" and path.exists() and path.stat().st_size > 0:
        with path.open("a", encoding="utf-8") as f:
            f.write("\n")
            
    # 正式写入数据
    with path.open(mode, encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare medical SFT data from raw JSON files.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/medical_raw"),
                        help="Raw data directory")
    parser.add_argument("--out-dir", type=Path, default=Path("data/medical"),
                        help="Output directory for processed data")
    args = parser.parse_args()

    finetune_dir = args.raw_dir / "finetune"
    train_path = finetune_dir / "train_zh_0.json"
    valid_path = finetune_dir / "valid_zh_0.json"
    test_path = finetune_dir / "test_zh_0.json"

    if not any(p.exists() for p in [train_path, valid_path, test_path]):
        print(f"原始数据不存在: {finetune_dir}")
        print("请先运行 python download_dataset.py 下载数据集")
        return
    
    print(f"读取原始文件: train_zh_0.json, valid_zh_0.json, test_zh_0.json")

    train_data, valid_data, test_data = [], [], []
    eval_prompts = []

    def process_file(path, sft_list, eval_list):
        for item in read_jsonl(path):
            sft_item, eval_item = process_item(item)
            if sft_item is not None:
                sft_list.append(sft_item)
            elif eval_item is not None:
                eval_list.append(eval_item)

    process_file(train_path, train_data, eval_prompts)
    process_file(valid_path, valid_data, eval_prompts)
    process_file(test_path, test_data, eval_prompts)

    print(f"清洗后: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    print(f"本次提取的评估问题 (eval_prompts): {len(eval_prompts)} 条")

    # --- 写出 jsonl 文件 ---
    # train/valid/test 每次重新生成，所以用默认的 'w' 覆盖模式
    write_jsonl(args.out_dir / "train.jsonl", train_data)
    write_jsonl(args.out_dir / "valid.jsonl", valid_data)
    write_jsonl(args.out_dir / "test.jsonl", test_data)
    
    # 核心修改：eval_prompts 使用 'a' 追加模式，保留原有数据！
    write_jsonl(args.out_dir / "eval_prompts.jsonl", eval_prompts, mode="a")


if __name__ == "__main__":
    main()