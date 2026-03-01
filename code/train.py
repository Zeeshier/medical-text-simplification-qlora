import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# ─── Configuration ────────────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
NEW_MODEL_NAME = "llama-3.2-1b-medical-simplifier"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HF_TOKEN = os.environ.get("HF_TOKEN")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY")

print(f"🔧 Device : {DEVICE}")
print(f"🔧 Visible GPUs: {torch.cuda.device_count()}")

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="llama3-medical-simplification",
        name="qlora-llama3.2-1b-plos",
    )

# ─── Load Model & Tokenizer ──────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, 
    token=HF_TOKEN, 
    trust_remote_code=False
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    token=HF_TOKEN,
    trust_remote_code=False,
)
model.config.use_cache = False

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)

import bitsandbytes as bnb
model.config.torch_dtype = torch.float16
for name, param in model.named_parameters():
    if param.dtype == torch.bfloat16:
        param.data = param.data.to(torch.float16)

# Force LoRA layers out of bfloat16 specifically
for name, module in model.named_modules():
    if 'lora_' in name:
        module.to(torch.float16)

model.print_trainable_parameters()

# ─── Dataset ─────────────────────────────────────────────────────────────
dataset = load_dataset("pszemraj/scientific_lay_summarisation-plos-norm")

SYSTEM_PROMPT = (
    "You are a medical text simplifier. Your job is to rewrite complex medical "
    "and scientific text so that a 5th grader can easily understand it. Use "
    "simple words, short sentences, and explain any technical terms. Keep the "
    "key facts accurate."
)

def format_chat(example):
    technical_text = example.get("article", example.get("input", ""))
    simple_text = example.get("summary", example.get("target", ""))
    
    max_input_chars = 1200
    if len(technical_text) > max_input_chars:
        technical_text = technical_text[:max_input_chars] + "..."
        
    chat = (
        "<|begin_of_text|>\n"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Simplify the following medical text:\n\n{technical_text}<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{simple_text}<|eot_id|>"
    )
    return {"text": chat}

train_dataset = dataset["train"].map(format_chat, remove_columns=dataset["train"].column_names)
eval_dataset = dataset["validation"].map(format_chat, remove_columns=dataset["validation"].column_names)

# ─── Training ────────────────────────────────────────────────────────────
sft_config = SFTConfig(
    output_dir="./results",
    max_steps=300,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=30,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    fp16=False,
    bf16=False,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataset_text_field="text",
    max_length=512,
    report_to="wandb" if WANDB_API_KEY else "none",
    run_name="qlora-llama3.2-1b-plos-fast",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.1,
    seed=42,
    ddp_find_unused_parameters=None,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8),
)

print("🚀 Starting training...")
trainer.train()

trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
print("✅ Training complete! Model saved to ./final_model")

if HF_TOKEN:
    print(f"📤 Pushing model to Hugging Face Hub as: {NEW_MODEL_NAME}")
    model.push_to_hub(NEW_MODEL_NAME, token=HF_TOKEN, private=True)
    tokenizer.push_to_hub(NEW_MODEL_NAME, token=HF_TOKEN)

if WANDB_API_KEY:
    wandb.finish()
