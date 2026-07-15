# Assignment 2A — Fine-Tuning Lab: LLaMA-3-8B Customer Support Assistant

## 1. Overview

This project fine-tunes **Meta-LLaMA-3-8B** (base, non-instruct) with **QLoRA** to act as a
customer support assistant, then deploys the resulting model behind a SageMaker real-time
endpoint, a Lambda + API Gateway layer, and a Streamlit chat UI.

| Item | Value |
|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B` |
| Method | QLoRA (4-bit NF4 quantization + LoRA adapters) |
| Task | Customer support Q&A / intent response generation |
| Training infra | AWS SageMaker training job, `ml.g5.2xlarge` |
| Inference infra | SageMaker real-time endpoint (HuggingFace TGI container), `ml.g5.4xlarge` |
| Serving stack | API Gateway → Lambda → SageMaker endpoint → Streamlit front end |

---

## 2. Dataset

- **Source:** [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (Hugging Face Hub)
- **Domain:** customer-service intents spanning ~20 industry verticals (retail, banking, healthcare, travel, insurance, etc.)
- **Size used:** first 15,000 examples of the full dataset, split 95/5 → **14,250 train / 750 eval** examples (`seed=42`)
- **Fields used:** `instruction` (customer message) → `response` (agent reply)

### Preprocessing pipeline (`chunking_dataset.ipynb`)

1. **Chat formatting** — Since LLaMA-3-8B is the *base* model (not `-Instruct`), it ships without
   a chat template. A LLaMA-3-style template was set manually on the tokenizer so every example is
   wrapped as:
   ```
   <|start_header_id|>system<|end_header_id|>
   You are a helpful and empathetic customer service assistant...<|eot_id|>
   <|start_header_id|>user<|end_header_id|>
   {instruction}<|eot_id|>
   <|start_header_id|>assistant<|end_header_id|>
   {response}<|eot_id|>
   ```
2. **Loss masking** — `format_and_mask()` builds the full prompt+response text *and* a prompt-only
   version, then masks every prompt token's label to `-100`. This ensures the model is only trained
   to predict the assistant's response tokens, not to re-generate the system/user turns.
3. **Packing** — Tokenized examples are concatenated and split into fixed **2048-token chunks**
   (`chunk()` function), carrying `input_ids`, `attention_mask`, and `labels` through together so
   masked positions stay aligned after packing. Leftover tokens are carried into the next batch via
   a `remainder` buffer, tracked separately for train and eval so eval doesn't leak train remainder.
4. Packed datasets are uploaded to S3 (`s3://llm-dataset-2026-tutorial/processed/llama/bitext/{train,eval}`)
   for the SageMaker training job to consume.

> ⚠️ **Security note before pushing to GitHub:** `chunking_dataset.ipynb` currently has a real
> Hugging Face access token hard-coded in a `login(token='hf_...')` call. Rotate that token and
> remove it from the notebook (use `huggingface-cli login`, an environment variable, or a
> SageMaker secret instead) before this repo goes public — an exposed token in git history is
> retrievable even after you delete it from the latest commit.

---

## 3. QLoRA Configuration (`run_llama.py`)

### Quantization (the "Q" in QLoRA)

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```
- **NF4** (NormalFloat4) quantization — the format QLoRA's authors found best preserves accuracy for
  normally-distributed weights, versus plain int4.
- **Double quantization** — quantizes the quantization constants themselves, saving ~0.4 bits/parameter
  with negligible quality loss. Necessary to fit an 8B model on a single `ml.g5.2xlarge` GPU (24GB A10G).
- **bf16 compute dtype** — the frozen 4-bit weights are de-quantized to bf16 on the fly for matmuls,
  keeping numerical stability while the adapters train in bf16.

### LoRA adapter config

```python
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=modules,       # auto-detected, see below
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
```

| Parameter | Value | Why |
|---|---|---|
| `r` (rank) | 16 | Standard starting point for 7–8B causal LMs — enough capacity to shift response style/tone without the adapter itself becoming a meaningful fraction of model size. Lower ranks (4–8) risk under-fitting a domain shift this large (base model → task-specific assistant persona); higher ranks (32–64) rarely help for a single, fairly narrow task and cost more VRAM/training time. |
| `lora_alpha` | 32 | Follows the common **alpha = 2 × r** convention, which keeps the effective learning-rate scaling (`alpha/r`) at a sensible default of 2 without needing to retune the base LR further. |
| `target_modules` | auto-detected via `find_all_linear_names()` | Rather than hand-picking `q_proj`/`v_proj` etc., the script scans the quantized model for every `bnb.nn.Linear4bit` layer and targets all of them (excluding `lm_head`, which stays frozen). This applies LoRA to *all* linear projections (attention **and** MLP blocks), which QLoRA's original paper found matches or beats attention-only targeting for causal LM fine-tuning. |
| `lora_dropout` | 0.05 | Light regularization — enough to reduce adapter overfitting on a single-epoch-scale run without meaningfully slowing convergence. |
| `bias` | `"none"` | Standard for LoRA — biases are left frozen/untouched since training them adds negligible expressiveness but doubles the number of tracked parameter groups. |
| `task_type` | `CAUSAL_LM` | Matches the LLaMA-3 architecture being fine-tuned. |

### Training arguments

| Parameter | Value | Notes |
|---|---|---|
| `per_device_train_batch_size` | 1 | VRAM-constrained on `ml.g5.2xlarge` (single A10G, 24GB) at 2048-token sequence length |
| `gradient_accumulation_steps` | 4 | Effective batch size = 1 × 4 = **4** |
| `epochs` | 2 | |
| `learning_rate` | 2e-4 | Typical for LoRA (adapters trained from scratch tolerate higher LR than full fine-tuning) |
| `optim` | `paged_adamw_8bit` | Paged optimizer states avoid OOM spikes during gradient accumulation, ~2GB lighter than standard AdamW |
| `bf16` | `True` (on Ampere+ GPUs) | |
| `gradient_checkpointing` | `True` | Trades compute for memory — necessary to fit 8B params + activations in 24GB |
| `save_strategy` / `save_total_limit` | epoch / 2 | |
| `evaluation_strategy` | epoch (when eval set provided) | |
| Merge behavior | `merge_weights=True` | LoRA adapters are merged back into the base weights (`merge_and_unload()`) after training, so the SageMaker endpoint serves a single dense model rather than base + adapter at inference time |

---

## 4. Training Job

Launched via SageMaker's `HuggingFace` estimator (`chunking_dataset.ipynb`):

- **Instance:** `ml.g5.2xlarge` (single A10G GPU, 24GB)
- **Framework versions:** `transformers==4.36.0`, `pytorch==2.1.0`, `py310`
- **Entry point:** `run_llama.py`
- **Hyperparameters passed:** `model_id=meta-llama/Meta-Llama-3-8B`, `epochs=2`, `per_device_train_batch_size=1`, `lr=2e-4`, `merge_weights=True`
- **Max runtime:** 7,200s (2 hours)
- **Output:** merged fp16 model artifact written to `s3://.../meta-Llama-3-8B-qlora-<timestamp>/output/model.tar.gz`

### Training loss & eval metrics

> **Not yet filled in** — the uploaded files don't include the CloudWatch training logs or the
> `trainer.evaluate()` output printed at the end of `run_llama.py`. To complete this section, pull
> the loss curve from either:
> - the SageMaker console → **Training jobs** → your job name → **Monitor** tab (loss is logged
>   every 10 steps per `logging_steps=10`), or
> - CloudWatch Logs group `/aws/sagemaker/TrainingJobs` for that job name.
>
> Paste the loss-vs-step values here (a simple table or a plotted PNG committed alongside this
> report works well), plus the `Final eval metrics: {...}` line that `run_llama.py` prints after
> `trainer.evaluate()`. Suggested structure:
>
> | Step | Train loss |
> |---|---|
> | 10 | ... |
> | ... | ... |
>
> | Eval metric | Value |
> |---|---|
> | eval_loss | ... |
> | eval_runtime | ... |

---

## 5. Base vs. Fine-Tuned Comparison (10 test prompts)

> **Not yet filled in** — the uploaded `deploying_model.ipynb` only shows one inference call against
> the *fine-tuned* endpoint (the "How do I make an order?" example below); there's no corresponding
> base-model-only run captured in the files provided, so a real comparison can't be written without
> fabricating outputs.
>
> To generate this section: stand up a second, cheap endpoint (or a local/notebook load) of the
> **unmodified** `meta-llama/Meta-Llama-3-8B` using the same prompt template but *without* the
> system prompt tuned into the assistant persona, run the same 10 customer-support prompts through
> both, and drop the outputs into the table below. Good candidate prompts to reuse from your
> Streamlit sidebar: *"What is your return policy?"*, *"How do I cancel an order?"*, *"How do I
> track my delivery?"*, *"I was charged twice, help!"*, *"How do I reset my password?"*, *"Where is
> my refund?"*, plus 4 more of your choice.
>
> | # | Prompt | Base model output | Fine-tuned output |
> |---|---|---|---|
> | 1 | | | |
> | 2 | | | |
> | ... | | | |
>
> One real example is available from the deployment notebook, for reference on the fine-tuned side:
>
> **Prompt:** *"How do I make an order?"*
> **Fine-tuned output:** a 6-step, numbered walkthrough (visit website → add to cart → checkout →
> enter shipping/contact info → select payment → confirm order), written in a polite, professional
> tone with no invented names/links — matching the system prompt's constraints exactly.

---

## 6. Observations

> **Draft below based on what's verifiable from the code/config; replace/extend once you have the
> real loss curve and 10-prompt comparison from Sections 4–5.**

1. **Format adherence held up well.** The one captured inference example shows the model
   respecting every hard constraint in the system prompt (no invented placeholders, single
   structured answer, no persona name) — evidence the loss-masking approach (training only on
   assistant tokens, not the template scaffolding) taught the model *when* to stop, not just *what*
   to say.
2. **The `stop` sequences do real work.** Because the base model was never instruction-tuned to
   respect turn boundaries, the TGI `stop` list (`<|eot_id|>`, all three `<|start_header_id|>...`
   variants) is doing a meaningful share of the "produce one answer only" behavior alongside the
   system prompt — worth calling out as a deployment-time safeguard, not purely a fine-tuning
   result.
3. **(Placeholder)** Add your third observation once you have eval-loss-vs-train-loss numbers —
   e.g. whether eval loss plateaued/diverged across the 2 epochs, which would indicate whether more
   epochs or a lower LR would help.

---

## 7. Deployment Architecture

```
Streamlit UI  →  API Gateway  →  AWS Lambda  →  SageMaker real-time endpoint (TGI container)
(streamlit_customerbotapp.py)   (invoke_lambda.py)     ("llama3endpoint--v1", ml.g5.4xlarge)
```

- **Model server:** Hugging Face TGI container (`huggingface-pytorch-tgi-inference:2.1.1-tgi1.3.3`),
  deployed via `HuggingFaceModel.deploy()` with `SM_NUM_GPUS=1`, `MAX_INPUT_LENGTH=2048`,
  `MAX_TOTAL_TOKENS=4096`, `MAX_BATCH_TOTAL_TOKENS=20480` (headroom for ~5 concurrent users at max length).
- **Lambda (`invoke_lambda.py`):** rebuilds the same LLaMA-3 chat-template prompt string used in
  training, calls `sagemaker-runtime.invoke_endpoint`, and returns `generated_text` (or a structured
  error) as API Gateway's JSON body.
- **Frontend (`streamlit_customerbotapp.py`):** dark-themed chat UI with suggested-question chips,
  session-scoped chat history, and a query counter; calls the API Gateway endpoint over HTTPS and
  unwraps the Lambda proxy-integration response shape.

### Inference parameters (shared by notebook test call and Lambda)

| Parameter | Value | Purpose |
|---|---|---|
| `temperature` | 0.2 | Low — favors deterministic, on-policy support answers over creative variation |
| `top_p` | 0.7 | |
| `top_k` | 40 | |
| `max_new_tokens` | 150 | Caps response length (~750 words max) |
| `repetition_penalty` | 1.12 | Discourages repeated phrases |
| `do_sample` | `False` | Greedy decoding — consistent with the low temperature choice for a support bot |

---

## 8. Repro Notes / Requirements

```
transformers==4.36.0
peft==0.7.1
bitsandbytes==0.43.0
datasets==4.8.2
huggingface_hub==0.36.2
accelerate==0.26.0
scipy==1.11.4
```

To reproduce: run `chunking_dataset.ipynb` end-to-end (dataset prep → S3 upload → SageMaker
training job) with `run_llama.py` as the entry point, then `deploying_model.ipynb` to deploy the
merged model artifact behind a SageMaker endpoint. `invoke_lambda.py` and
`streamlit_customerbotapp.py` complete the serving stack.
