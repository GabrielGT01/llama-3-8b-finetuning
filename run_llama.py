import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch
import bitsandbytes as bnb
from huggingface_hub import login, HfFolder


def str2bool(v):
    return str(v).lower() in ("yes", "true", "1")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_id", type=str,
                        help="model_id used for training")

    parser.add_argument("--hf_token", type=str,
                        default=None)           

    parser.add_argument("--dataset_path", type=str,
                        default="lm_dataset")

    parser.add_argument("--eval_dataset_path", type=str,
                        default=None)

    parser.add_argument("--epochs", type=int,
                        default=2)

    parser.add_argument("--per_device_train_batch_size",
                        type=int, default=1)

    parser.add_argument("--lr", type=float,
                        default=5e-5)

    parser.add_argument("--seed", type=int,
                        default=42)

    parser.add_argument("--gradient_checkpointing",
                        type=str2bool,          
                        default=True)

    parser.add_argument("--bf16",
                        type=str2bool,          
                        default=True if torch.cuda.get_device_capability()[0] == 8
                        else False)

    parser.add_argument("--merge_weights",
                        type=str2bool,         
                        default=True)

    args, _ = parser.parse_known_args()

    if args.hf_token:
        print("Logging into HuggingFace Hub")
        login(token=args.hf_token)

    return args


def print_trainable_parameters(model, use_4bit=False):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    if use_4bit:
        trainable_params /= 2
    print(
        f"all params: {all_param:,d} || "
        f"trainable params: {trainable_params:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(
                names[0] if len(names) == 1 else names[-1]
            )
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=gradient_checkpointing
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    # r=16 industry standard starting point 
    peft_config = LoraConfig(               
        r=16,
        lora_alpha=32,                      # r × 2 rule 
        target_modules=modules,
        lora_dropout=0.05,                  
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.print_trainable_parameters()
    return model


def training_function(args):
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)

    eval_dataset = None
    if args.eval_dataset_path:
        eval_dataset = load_from_disk(args.eval_dataset_path)
        print(f"Loaded eval dataset: {len(eval_dataset)} packed samples")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        use_cache=False if args.gradient_checkpointing else True,
        device_map={"": torch.cuda.current_device()},  #  pin to current GPU / auto
        quantization_config=bnb_config,
    )

    model = create_peft_model(
        model,
        gradient_checkpointing=args.gradient_checkpointing,
        bf16=args.bf16
    )

    output_dir = "/tmp/llama"            

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=4,   # add this — effective batch size = 1×4 = 4
        bf16=args.bf16,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        optim="paged_adamw_8bit",        # add this — saves ~2GB vs default adamw
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        print(f"Final eval metrics: {eval_metrics}")

    sagemaker_save_dir = "/opt/ml/model/"

    if args.merge_weights:
        trainer.model.save_pretrained(
            output_dir, safe_serialization=False
        )
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            output_dir,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        model = model.merge_and_unload()
        model.save_pretrained(
            sagemaker_save_dir,
            safe_serialization=True,
            max_shard_size="2GB"
        )
    else:
        trainer.model.save_pretrained(
            sagemaker_save_dir,
            safe_serialization=True
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained(sagemaker_save_dir)


def main():
    args = parse_args()                     
    training_function(args)


if __name__ == "__main__":
    main()