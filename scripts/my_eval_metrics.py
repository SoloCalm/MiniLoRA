""" 
python scripts/my_eval_metrics
python scripts/my_eval_metrics.py --max-samples 5
"""
import argparse
import csv
import gc
import json
import math
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM_PROMPT = "你是一名谨慎的医疗健康问答助手。回答应清晰、专业，并提醒用户必要时及时就医。"

USER_PROMPT_TEMPLATE = "请以谨慎、专业、易懂的方式回答下面的医疗健康问题：{question}"


def load_model(model_name, adapter_dir=None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir or model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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


def generate(tokenizer, model, question, max_new_tokens=256):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
        )

    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def load_jsonl(path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def char_tokens(text):
    return [char for char in str(text) if not char.isspace()]


def ngrams(tokens, n):
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(reference, prediction, max_n=4):
    ref_tokens = char_tokens(reference)
    pred_tokens = char_tokens(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_counts = Counter(ngrams(pred_tokens, n))
        ref_counts = Counter(ngrams(ref_tokens, n))
        if not pred_counts:
            # precisions.append(0.0)
            precisions.append(1e-10) # 避免 log(0) 导致整个 BLEU 变成 0，改为一个非常小的数
            continue
        overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        precisions.append((overlap + 1) / (sum(pred_counts.values()) + 1))

    log_precision = sum(math.log(precision) for precision in precisions) / max_n
    brevity_penalty = 1.0
    if len(pred_tokens) < len(ref_tokens):
        brevity_penalty = math.exp(1 - len(ref_tokens) / len(pred_tokens))
    return brevity_penalty * math.exp(log_precision)


def rouge_n(reference, prediction, n):
    ref_counts = Counter(ngrams(char_tokens(reference), n))
    pred_counts = Counter(ngrams(char_tokens(prediction), n))
    if not ref_counts or not pred_counts:
        return 0.0
    overlap = sum(min(count, pred_counts[gram]) for gram, count in ref_counts.items())
    return overlap / sum(ref_counts.values())


def lcs_length(a, b):
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(reference, prediction):
    ref_tokens = char_tokens(reference)
    pred_tokens = char_tokens(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, pred_tokens)
    recall = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)
    if recall + precision == 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def calc_scores(reference, prediction):
    return {
        "bleu1": bleu_score(reference, prediction, max_n=1),
        "bleu2": bleu_score(reference, prediction, max_n=2),
        "bleu4": bleu_score(reference, prediction, max_n=4),
        "rouge1": rouge_n(reference, prediction, n=1),
        "rouge2": rouge_n(reference, prediction, n=2),
        "rougeL": rouge_l(reference, prediction),
    }


def average_scores(score_list):
    if not score_list:
        return {}
    return {
        key: sum(scores[key] for scores in score_list) / len(score_list)
        for key in score_list[0]
    }


def write_summary_csv(path, base_scores, lora_scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ["bleu1", "bleu2", "bleu4", "rouge1", "rouge2", "rougeL"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", *metric_names])
        writer.writeheader()
        writer.writerow({"model": "base", **base_scores})
        writer.writerow({"model": "lora", **lora_scores})


def main():
    parser = argparse.ArgumentParser(description="Evaluate base vs LoRA with character BLEU/ROUGE.")
    parser.add_argument("--model-name", default="models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/qwen-medical-lora"))
    parser.add_argument("--eval-file", type=Path, default=Path("data/medical/test.jsonl"))
    parser.add_argument("--output-file", type=Path, default=Path("results/base_vs_lora_metrics.jsonl"))
    parser.add_argument("--summary-file", type=Path, default=Path("results/metrics_summary.csv"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.eval_file.exists():
        print(f"Eval file not found: {args.eval_file}")
        print('Expected jsonl records with fields like {"input": "...", "output": "..."}')
        return

    eval_items = load_jsonl(args.eval_file)
    examples = []
    for item in eval_items:
        question = str(item.get("input") or item.get("question") or item.get("instruction", "")).strip()
        reference = str(item.get("output", "")).strip()
        if question and reference:
            examples.append({"question": question, "reference": reference})

    if args.max_samples > 0:
        examples = examples[: args.max_samples]

    if not examples:
        print(f"No valid examples found in {args.eval_file}. Need non-empty question/input and output.")
        return

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loaded {len(examples)} evaluation examples from {args.eval_file}")

    print("\n[1/2] Loading base model...")
    tokenizer_base, model_base = load_model(args.model_name, adapter_dir=None)
    results = []
    for index, example in enumerate(tqdm(examples, desc="Base Generating"), start=1):
        print(f"  Base generating {index}/{len(examples)}")
        base_answer = generate(tokenizer_base, model_base, example["question"], args.max_new_tokens)
        results.append(
            {
                "question": example["question"],
                "reference": example["reference"],
                "base_answer": base_answer,
                "base_scores": calc_scores(example["reference"], base_answer),
            }
        )

    del model_base
    del tokenizer_base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.adapter_dir.exists():
        print(f"\nLoRA adapter not found: {args.adapter_dir}")
        print("Base results were generated, but LoRA scoring was skipped.")
        write_jsonl(args.output_file, results)
        base_avg = average_scores([item["base_scores"] for item in results])
        write_summary_csv(args.summary_file, base_avg, {key: 0.0 for key in base_avg})
        return

    print("\n[2/2] Loading LoRA model...")
    tokenizer_lora, model_lora = load_model(args.model_name, args.adapter_dir)
    for index, result in enumerate(tqdm(results, desc="LoRA Generating"), start=1):
        print(f"  LoRA generating {index}/{len(results)}")
        lora_answer = generate(tokenizer_lora, model_lora, result["question"], args.max_new_tokens)
        result["lora_answer"] = lora_answer
        result["lora_scores"] = calc_scores(result["reference"], lora_answer)

    base_avg = average_scores([item["base_scores"] for item in results])
    lora_avg = average_scores([item["lora_scores"] for item in results])

    write_jsonl(args.output_file, results)
    write_summary_csv(args.summary_file, base_avg, lora_avg)

    print(f"\nDetailed results saved to: {args.output_file}")
    print(f"Summary saved to: {args.summary_file}")
    print("\nAverage scores:")
    for model_name, scores in [("base", base_avg), ("lora", lora_avg)]:
        score_text = ", ".join(f"{key}={value:.4f}" for key, value in scores.items())
        print(f"  {model_name}: {score_text}")


if __name__ == "__main__":
    main()
