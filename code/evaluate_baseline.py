"""
evaluate_baseline.py — Baseline evaluation of the UNMODIFIED Llama-3.2-1B-Instruct model.

Runs the same ROUGE-L and Flesch-Kincaid evaluation as evaluate.py, but on the
base model WITHOUT fine-tuning. This provides the "before" comparison point.

Usage:
    python evaluate_baseline.py --num_samples 50
"""

import argparse
import os
import json
import numpy as np
import torch
import textstat
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from rouge_score import rouge_scorer

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_ID = "pszemraj/scientific_lay_summarisation-plos-norm"

SYSTEM_PROMPT = (
    "You are a medical text simplifier. Your job is to rewrite complex medical "
    "and scientific text so that a 5th grader can easily understand it. Use "
    "simple words, short sentences, and explain any technical terms. Keep the "
    "key facts accurate."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline evaluation of unmodified Llama 3.2")
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of test samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per sample")
    parser.add_argument("--output_file", type=str, default="baseline_results.json",
                        help="Output file for baseline results")
    return parser.parse_args()


def load_base_model():
    """Load the base Llama model WITHOUT any fine-tuning (4-bit for T4)."""
    hf_token = os.environ.get("HF_TOKEN", None)

    print(f"📦 Loading BASE model (no fine-tuning): {BASE_MODEL_ID}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        token=hf_token,
        trust_remote_code=False,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer


def generate_simplification(model, tokenizer, technical_text, max_new_tokens=512):
    """Generate a simplified version using the BASE model."""
    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Simplify the following medical text:\n\n{technical_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def evaluate_baseline(model, tokenizer, num_samples=20, max_new_tokens=512):
    """Run evaluation on the base model."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    raw_test = load_dataset(DATASET_ID, split="test")
    sample_indices = np.random.choice(len(raw_test), size=min(num_samples, len(raw_test)), replace=False)

    results = []
    rouge_l_scores = []
    fk_original_scores = []
    fk_simplified_scores = []

    print(f"\n📊 Baseline evaluation on {len(sample_indices)} test samples...\n")

    for i, idx in enumerate(sample_indices):
        example = raw_test[int(idx)]
        technical_text = example.get("article", example.get("input", ""))[:2000]
        reference = example.get("summary", example.get("target", ""))

        prediction = generate_simplification(model, tokenizer, technical_text, max_new_tokens)

        # ROUGE-L
        rouge_result = scorer.score(reference, prediction)
        rouge_l = rouge_result["rougeL"].fmeasure
        rouge_l_scores.append(rouge_l)

        # Flesch-Kincaid
        fk_orig = textstat.flesch_kincaid_grade(technical_text)
        fk_simplified = textstat.flesch_kincaid_grade(prediction) if len(prediction.split()) > 5 else float("nan")
        fk_original_scores.append(fk_orig)
        if not np.isnan(fk_simplified):
            fk_simplified_scores.append(fk_simplified)

        sample_result = {
            "index": int(idx),
            "rouge_l": rouge_l,
            "fk_original": fk_orig,
            "fk_simplified": fk_simplified if not np.isnan(fk_simplified) else None,
            "prediction_preview": prediction[:300],
            "reference_preview": reference[:300],
        }
        results.append(sample_result)

        if i < 5:
            print(f"── Example {i+1} ──")
            print(f"  📄 Original FK Grade  : {fk_orig:.1f}")
            print(f"  ✏️  Baseline FK Grade  : {fk_simplified:.1f}")
            print(f"  📏 ROUGE-L            : {rouge_l:.4f}")
            print(f"  📝 Output (first 200 chars): {prediction[:200]}...")
            print()

    # Aggregate
    avg_rouge_l = float(np.mean(rouge_l_scores))
    avg_fk_original = float(np.mean(fk_original_scores))
    avg_fk_simplified = float(np.mean(fk_simplified_scores)) if fk_simplified_scores else None
    fk_improvement = (avg_fk_original - avg_fk_simplified) if avg_fk_simplified else None

    summary = {
        "model": "BASELINE (no fine-tuning)",
        "num_samples": len(sample_indices),
        "rouge_l_avg": avg_rouge_l,
        "fk_original_avg": avg_fk_original,
        "fk_simplified_avg": avg_fk_simplified,
        "fk_improvement": fk_improvement,
        "samples": results,
    }

    print("\n" + "=" * 60)
    print("📊 BASELINE EVALUATION RESULTS")
    print("=" * 60)
    print(f"  ROUGE-L (avg)                    : {avg_rouge_l:.4f}")
    print(f"  Flesch-Kincaid Original (avg)     : {avg_fk_original:.1f}")
    print(f"  Flesch-Kincaid Baseline Out (avg)  : {avg_fk_simplified:.1f}" if avg_fk_simplified else "  Flesch-Kincaid Baseline Out (avg)  : N/A")
    print(f"  FK Grade Improvement              : {fk_improvement:+.1f} levels" if fk_improvement else "  FK Grade Improvement              : N/A")
    print("=" * 60)

    return summary


def main():
    args = parse_args()

    model, tokenizer = load_base_model()

    results = evaluate_baseline(model, tokenizer, num_samples=args.num_samples, max_new_tokens=args.max_new_tokens)

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Baseline results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
