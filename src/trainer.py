import torch
import json
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType
from .config import TrainConfig
from .data import load_viquad, extract_answer, format_chat_prompt


# ===========================
# PREPROCESS FUNCTION
# ===========================
def preprocess_function(example, tokenizer, cfg):
    context = example["context"]
    question = example["question"]
    answer = extract_answer(example)

    messages = format_chat_prompt(context, question, answer)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=cfg.max_seq_length,
        padding="max_length",
        pad_to_multiple_of=8,
        add_special_tokens=False,
    )

    messages_prompt_only = format_chat_prompt(context, question, answer=None)
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt_only,
        tokenize=False,
        add_generation_prompt=False
    )

    prompt_ids = tokenizer(
        prompt_text,
        truncation=True,
        max_length=cfg.max_seq_length,
        add_special_tokens=False
    )["input_ids"]

    input_ids = tokenized["input_ids"]
    prompt_len = len(prompt_ids)

    # Vectorized Masking
    labels = [
        -100 if i < prompt_len or token == tokenizer.pad_token_id else token
        for i, token in enumerate(input_ids)
    ]

    tokenized["labels"] = labels
    return tokenized


# ===========================
# TRAIN FUNCTION
# ===========================
def train():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    print(f"🚀 Loading tokenizer & model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto"
    )

    # Tối ưu hóa Model Config
    model.config.dropout = 0.0
    model.config.attention_dropout = 0.0

    # Xử lý Gradient Checkpointing (Nếu bật)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        # Nếu tắt checkpointing thì bật cache để inference/eval nhanh hơn chút
        model.config.use_cache = True

    if cfg.use_lora:
        target_mod = cfg.target_modules if cfg.target_modules else ["q_proj", "v_proj"]
        print(f"⚡ Applying LoRA on: {target_mod}")
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_mod
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    print(f"📂 Loading data from: {cfg.data_path}")
    raw_ds = load_viquad(cfg.data_path)

    print("🔄 Processing Train & Validation...")
    train_ds = raw_ds["train"].map(
        lambda x: preprocess_function(x, tokenizer, cfg),
        remove_columns=raw_ds["train"].column_names,
        desc="Mapping Train"
    )
    val_ds = raw_ds["validation"].map(
        lambda x: preprocess_function(x, tokenizer, cfg),
        remove_columns=raw_ds["validation"].column_names,
        desc="Mapping Validation"
    )

    print("🔄 Processing Test set for Loss Eval...")
    test_ds_for_loss = raw_ds["test"].map(
        lambda x: preprocess_function(x, tokenizer, cfg),
        remove_columns=raw_ds["test"].column_names,
        desc="Mapping Test (Loss)"
    )

    # --- TRAINING ARGUMENTS TỐI ƯU CHO RTX 3090 ---
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,

        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,

        logging_steps=cfg.logging_steps,
        eval_strategy=cfg.evaluation_strategy,
        eval_steps=cfg.eval_steps,
        save_strategy=cfg.save_strategy,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,

        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,

        # ⭐ CÁC TỐI ƯU QUAN TRỌNG ⭐
        dataloader_num_workers=4,  # Dùng 4 nhân CPU nạp data
        dataloader_pin_memory=True,  # Tăng tốc bắn data vào VRAM
        tf32=True,  # Bật TensorFloat-32 cho Ampere (RTX 30xx)
        optim="adamw_torch",  # Optimizer nhanh, nhẹ

        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=None,
    )

    print("🔥 Start Training...")
    trainer.train()

    print(f"💾 Saving adapter model to {cfg.output_dir}...")
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print("\n🧐 Calculating Test Loss...")
    test_metrics = trainer.evaluate(eval_dataset=test_ds_for_loss, metric_key_prefix="test")
    print("📊 TEST LOSS RESULT:", json.dumps(test_metrics, indent=4))

    with open(os.path.join(cfg.output_dir, "test_loss_results.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    train()