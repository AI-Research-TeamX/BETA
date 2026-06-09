"""
SFT Training on GameSolve-Bench Chain-of-Thought Solutions.
Uses DDP across 8 GPUs. Full fine-tuning on Qwen2.5-3B-Instruct.
"""
import os
import json
import math
import time
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        messages = entry["messages"]

        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_messages = messages[:2]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        full_enc = self.tokenizer(
            full_text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt", padding=False
        )
        prompt_enc = self.tokenizer(
            prompt_text, truncation=True, max_length=self.max_seq_len,
            return_tensors="pt", padding=False
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {"input_ids": input_ids, "labels": labels, "prompt_len": prompt_len}


def collate_fn(batch, pad_token_id):
    max_len = max(b["input_ids"].shape[0] for b in batch)
    batch_size = len(batch)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].shape[0]
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = 1

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--train_data", default="data/phase2_sft/train.json")
    p.add_argument("--val_data", default="data/phase2_sft/val.json")
    p.add_argument("--output_dir", default="results/phase2/sft_cot")
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--batch_size_per_gpu", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= 20:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            valid_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * valid_tokens
            total_tokens += valid_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


def main():
    args = parse_args()
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = (rank == 0)

    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"SFT Training: world_size={world_size}, device={device}")
        logger.info(f"Args: {vars(args)}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, trust_remote_code=True
    ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    train_ds = SFTDataset(args.train_data, tokenizer, args.max_seq_length)
    val_ds = SFTDataset(args.val_data, tokenizer, args.max_seq_length)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size_per_gpu, sampler=train_sampler,
        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size_per_gpu, sampler=val_sampler,
        collate_fn=lambda b: collate_fn(b, pad_id), num_workers=4, pin_memory=True,
    )

    steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        logger.info(f"Batches/epoch: {len(train_loader)}, Steps/epoch: {steps_per_epoch}, Total steps: {total_steps}")
        logger.info(f"Effective batch: {args.batch_size_per_gpu * world_size * args.gradient_accumulation_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

    global_step = 0
    best_val_loss = float("inf")
    train_log = []
    running_loss = 0.0
    running_tokens = 0

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            t0 = time.time()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            valid_tokens = (labels != -100).sum().item()
            running_loss += outputs.loss.item() * valid_tokens
            running_tokens += valid_tokens

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if is_main and global_step % args.log_steps == 0:
                    avg_loss = running_loss / max(running_tokens, 1)
                    ppl = math.exp(min(avg_loss, 20))
                    lr = scheduler.get_last_lr()[0]
                    step_time = time.time() - t0
                    logger.info(
                        f"[Epoch {epoch+1}/{args.num_epochs}] Step {global_step}/{total_steps}  "
                        f"loss={avg_loss:.4f}  ppl={ppl:.2f}  lr={lr:.2e}  time={step_time:.1f}s"
                    )
                    train_log.append({
                        "step": global_step, "epoch": epoch + 1,
                        "loss": avg_loss, "ppl": ppl, "lr": lr,
                    })
                    running_loss = 0.0
                    running_tokens = 0

                if is_main and global_step % args.eval_steps == 0:
                    val_loss = evaluate(model, val_loader, device)
                    val_ppl = math.exp(min(val_loss, 20))
                    logger.info(f"  [VAL] Step {global_step}: loss={val_loss:.4f}  ppl={val_ppl:.2f}")
                    train_log.append({
                        "step": global_step, "type": "val",
                        "val_loss": val_loss, "val_ppl": val_ppl,
                    })

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_model(model, tokenizer, args, "best")
                        logger.info(f"  New best val loss: {val_loss:.4f}")

                if is_main and global_step % args.save_steps == 0:
                    save_model(model, tokenizer, args, f"step_{global_step}")

    if is_main:
        save_model(model, tokenizer, args, "final")
        log_path = Path(args.output_dir) / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(train_log, f, indent=2)
        logger.info(f"SFT Training complete. Best val loss: {best_val_loss:.4f}")

    dist.destroy_process_group()


def save_model(model, tokenizer, args, tag):
    save_dir = Path(args.output_dir) / f"checkpoint-{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)
    inner_model = model.module if hasattr(model, 'module') else model
    inner_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    logger.info(f"Model saved to {save_dir}")


if __name__ == "__main__":
    main()
