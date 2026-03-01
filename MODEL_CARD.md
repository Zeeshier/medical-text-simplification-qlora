---
library_name: transformers
tags:
- medical
- simplification
- qlora
- llama-3
---

# Model Card for Llama-3.2-1B Medical Text Simplifier

<!-- Provide a quick summary of what the model is/does. -->
This model is a QLoRA fine-tuned version of Meta's `Llama-3.2-1B-Instruct`, specialized in medical text simplification. It is designed to rewrite complex biomedical and scientific abstracts into plain language that is readable at a 5th-grade level, while preserving factual accuracy.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
This is the model card of a 🤗 transformers model that has been fine-tuned on the Hub. This model leverages QLoRA (4-bit NF4 quantization + LoRA adapters) to adapt the base Llama-3.2-1B-Instruct model for translating complex medical terminology into easily understandable lay summaries. Only the LoRA adapter weights (~3.4M parameters) were updated during training, maintaining high efficiency.

- **Developed by:** Zeeshan Ahmad
- **Model type:** Causal Language Model (decoder-only transformer)
- **Language(s) (NLP):** English
- **License:** MIT (adapters) / Llama 3.2 Community License (base model)
- **Finetuned from model:** `meta-llama/Llama-3.2-1B-Instruct`

### Model Sources

<!-- Provide the basic links for the model. -->
- **Repository:** https://github.com/zeeshier/medical-text-simplification-qlora

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
- Simplifying medical research abstracts for patients and general audiences.
- Converting clinical reports or scientific literature into patient-friendly language.
- Aiding science communication by making specialized biomedical literature accessible to non-expert readers.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
- **Clinical decision support:** This model is NOT a medical device and should NOT be used for diagnosis, treatment, or medical advice under any circumstances.
- **Legal/regulatory documents:** Simplification may alter precise legal or clinical meaning.
- **Non-English text:** The model was trained exclusively on English medical texts.
- **Real-time clinical settings:** Not evaluated or cleared for clinical deployment.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
1. **Domain Bias:** Trained exclusively on PLOS journal articles. It may not generalize well to clinical notes, FDA documents, or other distinct medical text types.
2. **Context Truncation:** Source articles were truncated to 2,000 characters during training, meaning long-form papers may lose critical context.
3. **Factual Accuracy:** While the model is instructed to preserve facts, simplification inherently carries the risk of omitting important nuances or occasionally introducing inaccuracies.
4. **Capacity Limits:** With 1B parameters, it may not align as robustly as larger language models (7B+, 70B+).

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->
Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. Simplified texts should ideally be reviewed by healthcare professionals before patient-facing use. This tool is meant to aid readability and accessibility, but it is not a substitute for proper health literacy programs or professional medical consultation.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
adapter_id = "zeeshier/llama-3.2-1b-medical-simplifier"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model, adapter_id)
tokenizer = AutoTokenizer.from_pretrained(adapter_id)

prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical text simplifier. Your job is to rewrite complex medical and scientific text so that a 5th grader can easily understand it. Use simple words, short sentences, and explain any technical terms. Keep the key facts accurate.<|eot_id|><|start_header_id|>user<|end_header_id|>

Simplify the following medical text:

The pathogenesis of Type 2 diabetes involves insulin resistance...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->
The model was trained on the `pszemraj/scientific_lay_summarisation-plos-norm` dataset.
- **Source:** PLOS (Public Library of Science) open-access journals.
- **Format:** Paired technical abstracts and author-written lay summaries.
- **Splits:** ~24,773 training samples, ~1,376 validation samples.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

Documents were truncated to 2,000 characters to fit within context length limits and formatted using the specialized Llama 3 Chat Template (incorporating a customized system prompt for Medical Text Simplification).

#### Training Hyperparameters

- **Quantization:** 4-bit NF4 with double quantization (compute dtype `float16`)
- **LoRA Config:** Rank 16, Alpha 32, Dropout 0.05
- **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `o_proj`
- **Batch Size:** 2 per device with 4 gradient accumulation steps (effective = 8)
- **Learning Rate:** 2e-4 (Cosine decay scheduler, 10% warmup steps)
- **Optimizer:** Paged AdamW 8-bit
- **Max Sequence Length:** 1024 tokens
- **Gradient Checkpointing:** Enabled
- **Epochs:** 1

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->
Evaluated on the test split of the `pszemraj/scientific_lay_summarisation-plos-norm` dataset (~1,376 samples).

#### Metrics

- **ROUGE-L:** Measures semantic overlap with reference lay summaries.
- **Flesch-Kincaid Grade Level:** Measures the reading difficulty of the text (target is a 5th-grade reading level).

### Results

- **ROUGE-L:** 0.2379 (up from baseline 0.0372)
- **Flesch-Kincaid Original:** 15.7
- **Flesch-Kincaid Simplified:** 14.2 (down from baseline 97.8)
- **FK Grade Improvement:** +1.5 levels (up from baseline -82.1 levels)

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->
Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700). Training leverages highly efficient QLoRA.

- **Hardware Type:** 1x NVIDIA T4 GPU (16GB VRAM)
- **Hours used:** ~2-4 hours
- **Cloud Provider:** Kaggle
- **Carbon Emitted:** ~0.06-0.12 kg CO₂ eq. (U.S. grid average)

## Technical Specifications

### Model Architecture and Objective

- Architecture: LlamaForCausalLM (decoder-only transformer)
- Objectives: Next-token prediction for instruction following.

### Compute Infrastructure

#### Hardware

Trained on freely available, single-GPU compute resources (NVIDIA T4 16GB VRAM).

#### Software

Hugging Face `transformers`, `peft`, `trl`, and `bitsandbytes`.

## Citation

**BibTeX:**

```bibtex
@misc{llama3-medical-simplifier-2026,
  title={Medical Text Simplification with QLoRA Fine-Tuned Llama 3.2},
  author={Zeeshan Ahmad},
  year={2026},
  publisher={Hugging Face}
}
```

## More Information

For full methodology, baseline evaluations, and training scripts, please refer to the project's repository.
