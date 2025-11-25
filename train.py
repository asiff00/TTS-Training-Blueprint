"""
Training script for Qwen2.5-0.5B with SNAC
"""
import ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train.huggingface.transformers import prepare_trainer
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
import torch
import wandb
import math
from torch.optim.lr_scheduler import LambdaLR

print("[STAGE 1/6] Initializing Ray...")
ray.init(ignore_reinit_error=True)
print("[STAGE 1/6] Ray initialized successfully")

print("[STAGE 2/6] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer_snac")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"[STAGE 2/6] Tokenizer loaded (vocab size: {len(tokenizer)})")

print("[STAGE 3/6] Loading dataset...")
full = load_from_disk("lj_speech_snac_ready")
print(f"[STAGE 3/6] Dataset loaded ({len(full)} samples)")
print("[STAGE 3/6] Splitting dataset (95% train / 5% eval)...")
split = full.train_test_split(test_size=0.05, seed=42)
train_ds, eval_ds = split["train"], split["test"]
print(f"[STAGE 3/6] Dataset split complete: {len(train_ds)} train, {len(eval_ds)} eval samples")

def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        l = len(item["input_ids"])
        input_ids[i, :l] = torch.tensor(item["input_ids"])
        labels[i, :l] = torch.tensor(item["labels"])
        mask[i, :l] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": mask}

def train_func():
    learning_rate = 5e-5
    max_train_steps = 200
    lr_warmup_steps = 10
    weight_decay = 0.01
    logging_steps = 1
    eval_steps = 10
    save_steps = 10
    bf16 = True
    report_to = "wandb"
    dataloader_num_workers = 0
    disable_tqdm = False
    adam_beta1 = 0.95
    adam_beta2 = 0.90
    adam_eps = 1e-7

    print("[STAGE 4/6] Initializing W&B...")
    wandb.init(project="qwen-snac-tts", name="qwen2.5-0.5b-7tpf", config={
        "max_steps": max_train_steps, 
        "lr": learning_rate,
        "scheduler": "cosine_with_warmup"
    })
    print("[STAGE 4/6] W&B initialized")

    print("[STAGE 4/6] Loading model (Qwen2.5-0.5B)...")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16)
    print(f"[STAGE 4/6] Resizing token embeddings to {len(tokenizer)}...")
    model.resize_token_embeddings(len(tokenizer))
    print("[STAGE 4/6] Model loaded and configured")

    print("[STAGE 5/6] Setting up optimizer and scheduler...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=weight_decay
    )

    def lr_decay_lambda(step):
        if step < lr_warmup_steps:
            return float(step) / float(max(1, lr_warmup_steps))
        progress = float(step - lr_warmup_steps) / float(max(1, max_train_steps - lr_warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_decay_lambda)
    print("[STAGE 5/6] Optimizer and scheduler configured")

    args = TrainingArguments(
        output_dir="./qwen-snac-tts",
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        max_steps=max_train_steps,
        bf16=bf16,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=logging_steps,
        report_to=report_to,
        dataloader_num_workers=dataloader_num_workers,
        disable_tqdm=disable_tqdm,
        remove_unused_columns=False,
        logging_first_step=True,
    )

    print("[STAGE 5/6] Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        optimizers=(optimizer, scheduler)
    )
    print("[STAGE 5/6] Trainer created")
    
    print("[STAGE 5/6] Preparing trainer for distributed training...")
    trainer = prepare_trainer(trainer)
    print("[STAGE 5/6] Trainer prepared")
    
    print("[STAGE 6/6] Starting training...")
    trainer.train()
    print("[STAGE 6/6] Training completed successfully!")
    wandb.finish()

print("\n" + "="*60)
print("Starting Ray TorchTrainer")
print("="*60 + "\n")

TorchTrainer(
    train_loop_per_worker=train_func,
    scaling_config=ScalingConfig(num_workers=1, use_gpu=True),
).fit()

print("\n" + "="*60)

print("✓ Training finished — check wandb for results")
print("="*60)
