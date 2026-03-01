import argparse
import os
import json
import numpy as np
import torch
import textstat
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from rouge_score import rouge_scorer

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_ID = "pszemraj/scientific_lay_summarisation-plos-norm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = (
    "You are a medical text simplifier. Your job is to rewrite complex medical "
    "and scientific text so that a 5th grader can easily understand it. Use "
    "simple words, short sentences, and explain any technical terms. Keep the "
    "key facts accurate."
)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned LoRA adapters")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of test samples")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--output_file", type=str, default="eval_results.json", help="Output file")
    parser.add_argument("--use_4bit", action="store_true", default=True, help="Use 4-bit quantization")
    return parser.parse_args()

def load_model(model_path, use_4bit=True):
    hf_token = os.environ.get("HF_TOKEN", None)
    print(f"📦 Loading base model: {BASE_MODEL_ID}")
    
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": {"": 0},
        "token": hf_token,
        "trust_remote_code": False,
    }
    
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, **model_kwargs)
    print(f"🔗 Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path, token=hf_token)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return model, tokenizer

def generate_simplification(model, tokenizer, technical_text, max_new_tokens):
    prompt = (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Simplify the following medical text:\n\n{technical_text}<|eot_id|>\n"
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

def evaluate(model, tokenizer, num_samples, max_new_tokens):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    raw_test = load_dataset(DATASET_ID, split="test")
    sample_indices = np.random.choice(len(raw_test), size=min(num_samples, len(raw_test)), replace=False)
    
    results = []
    rouge_l_scores = []
    fk_original_scores = []
    fk_simplified_scores = []
    
    print(f"\n📊 Evaluating on {len(sample_indices)} test samples...")
    for i, idx in enumerate(sample_indices):
        example = raw_test[int(idx)]
        technical_text = example.get("article", example.get("input", ""))[:2000]
        reference = example.get("summary", example.get("target", ""))
        
        prediction = generate_simplification(model, tokenizer, technical_text, max_new_tokens)
        
        rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure
        rouge_l_scores.append(rouge_l)
        
        fk_orig = textstat.flesch_kincaid_grade(technical_text)
        fk_simplified = textstat.flesch_kincaid_grade(prediction) if len(prediction.split()) > 5 else float("nan")
        fk_original_scores.append(fk_orig)
        if not np.isnan(fk_simplified):
            fk_simplified_scores.append(fk_simplified)
            
        results.append({
            "index": int(idx),
            "rouge_l": rouge_l,
            "fk_original": fk_orig,
            "fk_simplified": fk_simplified if not np.isnan(fk_simplified) else None,
        })
        
        if i < 3:
            print(f"── Example {i+1} ──")
            print(f"  📄 Original FK Grade  : {fk_orig:.1f}")
            print(f"  ✏️  Simplified FK Grade: {fk_simplified:.1f}")
            print(f"  📏 ROUGE-L            : {rouge_l:.4f}")
            print(f"  📝 Prediction (first 200 chars): {prediction[:200]}...")
            print()
            
    avg_rouge_l = float(np.mean(rouge_l_scores))
    avg_fk_original = float(np.mean(fk_original_scores))
    avg_fk_simplified = float(np.mean(fk_simplified_scores)) if fk_simplified_scores else None
    fk_improvement = (avg_fk_original - avg_fk_simplified) if avg_fk_simplified else None
    
    print("=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  ROUGE-L (avg)                    : {avg_rouge_l:.4f}")
    print(f"  Flesch-Kincaid Original (avg)     : {avg_fk_original:.1f}")
    if avg_fk_simplified:
        print(f"  Flesch-Kincaid Simplified (avg)   : {avg_fk_simplified:.1f}")
    if fk_improvement:
        print(f"  FK Grade Improvement              : {fk_improvement:+.1f} levels")
    print("=" * 60)
    
    return {"num_samples": len(sample_indices), "rouge_l_avg": avg_rouge_l, "fk_original_avg": avg_fk_original, "fk_simplified_avg": avg_fk_simplified, "samples": results}

def catastrophic_forgetting_check(model, tokenizer):
    general_questions = [
        "What is the capital of France?",
        "Explain photosynthesis in one sentence.",
        "Who wrote the play 'Romeo and Juliet'?",
        "What is 15 multiplied by 7?",
        "Name three planets in our solar system.",
    ]
    
    print("\n" + "=" * 60)
    print("🛡️  CATASTROPHIC FORGETTING CHECK")
    print("=" * 60)
    
    for q in general_questions:
        prompt = f"<|begin_of_text|>\n<|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=100, temperature=0.3, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        print(f"❓ Q: {q}\n💬 A: {answer}\n")

def main():
    args = parse_args()
    model, tokenizer = load_model(args.model_path, use_4bit=args.use_4bit)
    
    results = evaluate(model, tokenizer, num_samples=args.num_samples, max_new_tokens=args.max_new_tokens)
    catastrophic_forgetting_check(model, tokenizer)
    
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
