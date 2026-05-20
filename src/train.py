	import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from config import (
    MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, TARGET_MODULES,
    OUTPUT_DIR, NUM_EPOCHS, BATCH_SIZE, GRAD_ACCUM, LR
)

def main(data_path: str = "./data/sample_sft.jsonl"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # base load
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    dataset = load_dataset("json", data_files=data_path, split="train")

    # A100-safe: BF16 on, FP16 off
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        bf16=True,
        fp16=False,
        max_grad_norm=1.0,
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"✅ Saved adapter to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
