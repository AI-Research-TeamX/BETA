"""
Phase 2: GRPO + Auxiliary Probing Training for GameSolve-Bench
==============================================================
Full backbone fine-tuning + linear probing heads at layer L/2 (=18, 0-indexed 17).
Uses DDP across 8 GPUs. Mean pooling over prompt tokens for probing.

Usage:
    torchrun --nproc_per_node=8 train_grpo_probe.py [options]
"""

import os
import sys
import json
import math
import time
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from gamesolve_reward import compute_reward

logger = logging.getLogger(__name__)

# ── Probing head specs ────────────────────────────────────────────

LABEL_SPECS = {
    "eq_type":       {"num_classes": 3, "task_filter": "nash_equilibrium"},
    "difficulty":    {"num_classes": 3, "task_filter": None},
    "dominance":     {"num_classes": 2, "task_filter": None},
    "br_direction":  {"num_classes": 5, "task_filter": "best_response"},
    "eq_uniqueness": {"num_classes": 2, "task_filter": "nash_equilibrium"},
}

CONCEPT_WEIGHTS = {
    "br_direction":  1.57,
    "eq_uniqueness": 1.37,
    "eq_type":       1.34,
    "dominance":     1.23,
    "difficulty":    1.11,
}

SYSTEM_PROMPT = """You are an expert in game theory. Analyze the given game carefully and provide precise answers.

Always end your response with a clearly marked ANSWER section using this exact format:

For Nash Equilibrium tasks:
ANSWER:
Pure NE: [(action_row, action_col), ...]   or "none"
Mixed NE: [(sigma_row, sigma_col), ...]    or "none"
(sigma = probability vector, e.g. [0.4, 0.6])

For Best Response tasks:
ANSWER:
Best response action(s): [action_name, ...]
Expected payoffs: [eu_a1, eu_a2, ...]
Best response value: <number>
"""


# ── Dataset ───────────────────────────────────────────────────────

class GameSolveDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Probing Heads ─────────────────────────────────────────────────

class ProbingHeads(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, spec["num_classes"])
            for name, spec in LABEL_SPECS.items()
        })

    def forward(self, hidden_states, labels_dict):
        """
        hidden_states: (batch, hidden_size) - mean-pooled prompt representations
        labels_dict: {name: (batch,)} with -1 for N/A samples
        Returns: (total_loss, metrics_dict)
        """
        total_loss = torch.tensor(0.0, device=hidden_states.device)
        total_weight = 0.0
        metrics = {}
        for name, head in self.heads.items():
            labels = labels_dict[name]
            valid = labels >= 0
            if not valid.any():
                continue
            logits = head(hidden_states[valid])
            loss = F.cross_entropy(logits, labels[valid])
            w = CONCEPT_WEIGHTS.get(name, 1.0)
            total_loss = total_loss + loss * w
            total_weight += w
            with torch.no_grad():
                acc = (logits.argmax(-1) == labels[valid]).float().mean()
                metrics[f"probe/{name}_acc"] = acc.item()
                metrics[f"probe/{name}_loss"] = loss.item()
        if total_weight > 0:
            total_loss = total_loss / total_weight
        metrics["probe/total_loss"] = total_loss.item()
        return total_loss, metrics


# ── Hidden state capture ─────────────────────────────────────────

class HiddenStateCapture:
    """Register a forward hook on a specific layer to capture hidden states."""

    def __init__(self, model, layer_idx):
        self.hidden_states = None
        self._hook = model.model.layers[layer_idx].register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.hidden_states = output[0]

    def get(self):
        return self.hidden_states

    def clear(self):
        self.hidden_states = None

    def remove(self):
        self._hook.remove()


# ── GRPO utilities ────────────────────────────────────────────────

def compute_grpo_advantages(rewards_per_prompt, eps=1e-8):
    """
    rewards_per_prompt: list of lists, each inner list has N rewards for one prompt
    Returns: flat tensor of advantages (batch * N,)
    """
    all_advs = []
    for group in rewards_per_prompt:
        group_t = torch.tensor(group, dtype=torch.float32)
        mean_r = group_t.mean()
        std_r = group_t.std()
        advs = (group_t - mean_r) / (std_r + eps)
        all_advs.append(advs)
    return torch.cat(all_advs, dim=0)


def compute_log_probs_from_logits(logits, labels, mask):
    """
    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len) - token IDs
    mask: (batch, seq_len) - 1 for response tokens, 0 for prompt/pad
    Returns: per-token log probs (batch, seq_len)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return token_log_probs * mask


def mean_pool_prompt(hidden_states, prompt_mask):
    """
    hidden_states: (batch, seq_len, hidden_size)
    prompt_mask: (batch, seq_len) - 1 for prompt tokens, 0 for response/pad
    Returns: (batch, hidden_size)
    """
    mask_expanded = prompt_mask.unsqueeze(-1).float()
    summed = (hidden_states * mask_expanded).sum(dim=1)
    counts = mask_expanded.sum(dim=1).clamp(min=1)
    return summed / counts


# ── Generation ────────────────────────────────────────────────────

@torch.no_grad()
def generate_responses(model, tokenizer, prompt_ids, prompt_mask, n_rollouts,
                       max_new_tokens, temperature, top_p, device):
    """
    Generate n_rollouts responses per prompt.
    Returns: list of (n_prompts * n_rollouts) response strings
    """
    model.eval()
    n_prompts = prompt_ids.shape[0]
    all_responses = []
    all_response_ids = []

    for i in range(n_prompts):
        single_ids = prompt_ids[i:i+1]
        single_mask = prompt_mask[i:i+1]
        seq_len = single_mask.sum().item()
        single_ids = single_ids[:, :seq_len].to(device)
        single_mask = single_mask[:, :seq_len].to(device)

        expanded_ids = single_ids.expand(n_rollouts, -1)
        expanded_mask = single_mask.expand(n_rollouts, -1)

        outputs = model.generate(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = single_ids.shape[1]
        for j in range(n_rollouts):
            resp_ids = outputs[j, prompt_len:]
            resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
            all_responses.append(resp_text)
            all_response_ids.append(resp_ids.cpu())

    model.train()
    return all_responses, all_response_ids


# ── Training step ─────────────────────────────────────────────────

def build_training_batch(prompt_ids, prompt_mask, response_ids_list,
                         tokenizer, max_seq_len, device):
    """
    Build padded (prompt + response) sequences for training forward pass.
    Returns: input_ids, attention_mask, response_mask, prompt_only_mask
    """
    batch_input_ids = []
    batch_attn_mask = []
    batch_resp_mask = []
    batch_prompt_mask = []

    for i, resp_ids in enumerate(response_ids_list):
        prompt_idx = i // len(response_ids_list) * 0  # will be set properly by caller
        p_ids = prompt_ids[i]
        p_mask = prompt_mask[i]
        p_len = p_mask.sum().item()
        p_ids_trimmed = p_ids[:p_len]

        r_ids = resp_ids
        r_len = len(r_ids)

        total_len = p_len + r_len
        if total_len > max_seq_len:
            r_ids = r_ids[:max_seq_len - p_len]
            r_len = len(r_ids)
            total_len = p_len + r_len

        full_ids = torch.cat([p_ids_trimmed, r_ids.to(p_ids_trimmed.device)])
        attn = torch.ones(total_len, dtype=torch.long)
        resp = torch.zeros(total_len, dtype=torch.long)
        resp[p_len:] = 1
        pmask = torch.zeros(total_len, dtype=torch.long)
        pmask[:p_len] = 1

        batch_input_ids.append(full_ids)
        batch_attn_mask.append(attn)
        batch_resp_mask.append(resp)
        batch_prompt_mask.append(pmask)

    max_len = max(ids.shape[0] for ids in batch_input_ids)
    pad_id = tokenizer.pad_token_id

    padded_ids = torch.full((len(batch_input_ids), max_len), pad_id, dtype=torch.long)
    padded_attn = torch.zeros(len(batch_input_ids), max_len, dtype=torch.long)
    padded_resp = torch.zeros(len(batch_input_ids), max_len, dtype=torch.long)
    padded_prompt = torch.zeros(len(batch_input_ids), max_len, dtype=torch.long)

    for i in range(len(batch_input_ids)):
        l = batch_input_ids[i].shape[0]
        padded_ids[i, :l] = batch_input_ids[i]
        padded_attn[i, :l] = batch_attn_mask[i]
        padded_resp[i, :l] = batch_resp_mask[i]
        padded_prompt[i, :l] = batch_prompt_mask[i]

    return (padded_ids.to(device), padded_attn.to(device),
            padded_resp.to(device), padded_prompt.to(device))


def training_step(model, probing_heads, hidden_capture, tokenizer,
                  prompt_ids, prompt_mask, concept_labels, gt_info,
                  args, device, step_num):
    """
    One GRPO training step:
    1. Generate N responses per prompt
    2. Score with reward function
    3. Compute advantages
    4. Forward pass to get log_probs + hidden states
    5. Compute policy gradient loss + probing loss
    Returns: (total_loss, metrics_dict)
    """
    n_prompts = prompt_ids.shape[0]
    n_rollouts = args.n_rollouts
    metrics = {}

    # ── 1. Generate responses ──
    inner_model = model.module if hasattr(model, 'module') else model
    responses, response_ids = generate_responses(
        inner_model, tokenizer, prompt_ids, prompt_mask,
        n_rollouts=n_rollouts,
        max_new_tokens=args.max_response_length,
        temperature=args.gen_temperature,
        top_p=args.gen_top_p,
        device=device,
    )

    # ── 2. Score responses ──
    rewards_per_prompt = []
    all_rewards = []
    for i in range(n_prompts):
        group_rewards = []
        for j in range(n_rollouts):
            idx = i * n_rollouts + j
            r = compute_reward(
                responses[idx],
                gt_info[i]["ground_truth"],
                gt_info[i]["task"],
                gt_info[i]["row_labels"],
                gt_info[i]["col_labels"],
            )
            group_rewards.append(r)
        rewards_per_prompt.append(group_rewards)
        all_rewards.extend(group_rewards)

    metrics["reward/mean"] = np.mean(all_rewards)
    metrics["reward/std"] = np.std(all_rewards)
    metrics["reward/max"] = np.max(all_rewards)
    metrics["reward/min"] = np.min(all_rewards)

    # ── 3. Compute GRPO advantages ──
    advantages = compute_grpo_advantages(rewards_per_prompt)
    advantages = advantages.to(device)

    # ── 4. Build training sequences ──
    expanded_prompt_ids = prompt_ids.repeat_interleave(n_rollouts, dim=0)
    expanded_prompt_mask = prompt_mask.repeat_interleave(n_rollouts, dim=0)

    input_ids, attn_mask, resp_mask, prompt_only_mask = build_training_batch(
        expanded_prompt_ids, expanded_prompt_mask, response_ids,
        tokenizer, args.max_seq_length, device,
    )

    # ── 5. Forward pass with gradient accumulation ──
    total_n = input_ids.shape[0]
    micro_bs = args.micro_batch_size
    n_micro = math.ceil(total_n / micro_bs)

    accumulated_pg_loss = torch.tensor(0.0, device=device)
    accumulated_probe_loss = torch.tensor(0.0, device=device)
    all_probe_metrics = defaultdict(list)
    all_prompt_hidden = []

    model.train()
    hidden_capture.clear()

    for mi in range(n_micro):
        start = mi * micro_bs
        end = min(start + micro_bs, total_n)

        mb_ids = input_ids[start:end]
        mb_attn = attn_mask[start:end]
        mb_resp = resp_mask[start:end]
        mb_prompt = prompt_only_mask[start:end]
        mb_adv = advantages[start:end]

        hidden_capture.clear()
        outputs = model(input_ids=mb_ids, attention_mask=mb_attn, use_cache=False)
        logits = outputs.logits

        shift_logits = logits[:, :-1, :]
        shift_labels = mb_ids[:, 1:]
        shift_resp_mask = mb_resp[:, 1:]

        token_log_probs = compute_log_probs_from_logits(shift_logits, shift_labels, shift_resp_mask)
        resp_lengths = shift_resp_mask.sum(dim=-1).clamp(min=1)
        seq_mean_log_probs = token_log_probs.sum(dim=-1) / resp_lengths

        pg_loss = -(mb_adv * seq_mean_log_probs).mean()
        accumulated_pg_loss += pg_loss.detach()

        micro_loss = pg_loss / n_micro

        # Probing loss (use hidden states from first response per prompt)
        if args.probe_lambda > 0 and hidden_capture.get() is not None:
            h = hidden_capture.get()
            prompt_repr = mean_pool_prompt(h, mb_prompt[:, :h.shape[1]].float())

            prompt_indices = []
            for idx_in_mb in range(end - start):
                global_idx = start + idx_in_mb
                rollout_idx = global_idx % n_rollouts
                if rollout_idx == 0:
                    prompt_indices.append(idx_in_mb)

            if prompt_indices:
                probe_repr = prompt_repr[prompt_indices]
                mb_concept = {}
                for name in LABEL_SPECS:
                    p_indices_global = [(start + pi) // n_rollouts for pi in prompt_indices]
                    mb_concept[name] = concept_labels[name][p_indices_global].to(device)

                p_loss, p_metrics = probing_heads(probe_repr, mb_concept)
                micro_loss = micro_loss + (p_loss * args.probe_lambda) / n_micro
                accumulated_probe_loss += p_loss.detach()
                for k, v in p_metrics.items():
                    all_probe_metrics[k].append(v)

        micro_loss.backward()

    metrics["loss/pg"] = accumulated_pg_loss.item() / n_micro
    metrics["loss/probe"] = accumulated_probe_loss.item() / max(n_micro, 1)
    metrics["loss/total"] = metrics["loss/pg"] + args.probe_lambda * metrics["loss/probe"]

    for k, vals in all_probe_metrics.items():
        metrics[k] = np.mean(vals)

    return metrics


# ── Main ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="./Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--train_data", default="data/phase2/train.json")
    p.add_argument("--val_data", default="data/phase2/val.json")
    p.add_argument("--output_dir", default="results/phase2/full_probe_grpo")
    p.add_argument("--probe_layer", type=int, default=17, help="0-indexed layer for probing (L/2=18 → index 17)")
    p.add_argument("--n_rollouts", type=int, default=5)
    p.add_argument("--batch_size_per_gpu", type=int, default=2)
    p.add_argument("--micro_batch_size", type=int, default=4)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--max_response_length", type=int, default=512)
    p.add_argument("--max_seq_length", type=int, default=1536)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--probe_lr", type=float, default=1e-3)
    p.add_argument("--probe_lambda", type=float, default=0.1)
    p.add_argument("--gen_temperature", type=float, default=0.7)
    p.add_argument("--gen_top_p", type=float, default=0.9)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--log_steps", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


def set_seed(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def collate_fn(batch, tokenizer, max_prompt_len):
    prompts = []
    for sample in batch:
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sample["prompt"]},
        ]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    encoded = tokenizer(
        prompts, padding=True, truncation=True,
        max_length=max_prompt_len, return_tensors="pt",
    )

    concept_labels = {}
    for name in LABEL_SPECS:
        vals = [s["concept_labels"][name] for s in batch]
        concept_labels[name] = torch.tensor(vals, dtype=torch.long)

    gt_info = [{
        "ground_truth": s["ground_truth"],
        "task": s["task"],
        "row_labels": s["row_labels"],
        "col_labels": s["col_labels"],
    } for s in batch]

    return encoded["input_ids"], encoded["attention_mask"], concept_labels, gt_info


@torch.no_grad()
def evaluate(model, probing_heads, hidden_capture, tokenizer,
             val_loader, args, device, n_rollouts=2):
    """Run evaluation on validation set with fewer rollouts."""
    inner_model = model.module if hasattr(model, 'module') else model
    inner_model.eval()

    all_rewards = []
    all_probe_metrics = defaultdict(list)

    for batch_idx, (prompt_ids, prompt_mask, concept_labels, gt_info) in enumerate(val_loader):
        if batch_idx >= 10:
            break

        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        n_prompts = prompt_ids.shape[0]

        responses, response_ids = generate_responses(
            inner_model, tokenizer, prompt_ids, prompt_mask,
            n_rollouts=n_rollouts,
            max_new_tokens=args.max_response_length,
            temperature=0.1, top_p=0.95, device=device,
        )

        for i in range(n_prompts):
            for j in range(n_rollouts):
                idx = i * n_rollouts + j
                r = compute_reward(
                    responses[idx], gt_info[i]["ground_truth"],
                    gt_info[i]["task"], gt_info[i]["row_labels"],
                    gt_info[i]["col_labels"],
                )
                all_rewards.append(r)

        # Probe eval on prompt-only forward
        hidden_capture.clear()
        outputs = inner_model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=False)
        h = hidden_capture.get()
        if h is not None:
            prompt_repr = mean_pool_prompt(h, prompt_mask[:, :h.shape[1]].float())
            cl_device = {k: v.to(device) for k, v in concept_labels.items()}
            _, p_metrics = probing_heads(prompt_repr, cl_device)
            for k, v in p_metrics.items():
                all_probe_metrics[k].append(v)

    inner_model.train()
    metrics = {"val/reward_mean": np.mean(all_rewards) if all_rewards else 0.0}
    for k, vals in all_probe_metrics.items():
        metrics[f"val/{k}"] = np.mean(vals)
    return metrics


def main():
    args = parse_args()
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    set_seed(args.seed, rank)
    is_main = (rank == 0)

    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"World size: {world_size}, Device: {device}")
        logger.info(f"Args: {vars(args)}")

    # ── Load model ──
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, trust_remote_code=True,
    ).to(device)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    hidden_capture = HiddenStateCapture(model, args.probe_layer)

    # Probing heads
    probing_heads = ProbingHeads(model.config.hidden_size).to(device)

    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizers ──
    backbone_params = list(model.parameters())
    probe_params = list(probing_heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": probe_params, "lr": args.probe_lr, "weight_decay": 0.0},
    ])

    # ── Data ──
    train_ds = GameSolveDataset(args.train_data)
    val_ds = GameSolveDataset(args.val_data)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)

    _tokenizer = tokenizer
    _max_prompt_len = args.max_prompt_length

    def _collate(batch):
        return collate_fn(batch, _tokenizer, _max_prompt_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size_per_gpu, sampler=train_sampler,
        collate_fn=_collate, num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size_per_gpu, sampler=DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False),
        collate_fn=_collate, num_workers=2, pin_memory=True,
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if is_main:
        logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
        logger.info(f"Steps per epoch: {len(train_loader)}, Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")

    # ── Training loop ──
    global_step = 0
    best_val_reward = -1.0
    train_log = []

    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        epoch_metrics = defaultdict(list)

        for batch_idx, (prompt_ids, prompt_mask, concept_labels, gt_info) in enumerate(train_loader):
            t0 = time.time()
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)

            optimizer.zero_grad()

            step_metrics = training_step(
                model, probing_heads, hidden_capture, tokenizer,
                prompt_ids, prompt_mask, concept_labels, gt_info,
                args, device, global_step,
            )

            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(probing_heads.parameters()),
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()

            step_time = time.time() - t0
            step_metrics["time/step"] = step_time
            step_metrics["lr"] = scheduler.get_last_lr()[0]

            for k, v in step_metrics.items():
                epoch_metrics[k].append(v)

            global_step += 1

            if is_main and global_step % args.log_steps == 0:
                avg_metrics = {k: np.mean(v[-args.log_steps:]) for k, v in epoch_metrics.items()}
                log_str = f"[Epoch {epoch+1}/{args.num_epochs}] Step {global_step}/{total_steps}"
                for k in ["reward/mean", "loss/pg", "loss/probe", "loss/total", "lr", "time/step"]:
                    if k in avg_metrics:
                        v = avg_metrics[k]
                        fmt = f"{v:.2e}" if k == "lr" else f"{v:.4f}"
                        log_str += f"  {k}={fmt}"
                logger.info(log_str)

                # Probe accuracies
                probe_str = "  Probe:"
                for name in LABEL_SPECS:
                    key = f"probe/{name}_acc"
                    if key in avg_metrics:
                        probe_str += f" {name}={avg_metrics[key]:.3f}"
                logger.info(probe_str)

                train_log.append({
                    "step": global_step, "epoch": epoch + 1,
                    **{k: float(np.mean(v[-args.log_steps:])) for k, v in epoch_metrics.items()}
                })

            if is_main and global_step % args.eval_steps == 0:
                val_metrics = evaluate(
                    model, probing_heads, hidden_capture, tokenizer,
                    val_loader, args, device,
                )
                val_str = f"  [VAL] Step {global_step}:"
                for k, v in sorted(val_metrics.items()):
                    val_str += f" {k}={v:.4f}"
                logger.info(val_str)

                if val_metrics.get("val/reward_mean", 0) > best_val_reward:
                    best_val_reward = val_metrics["val/reward_mean"]
                    save_checkpoint(model, probing_heads, optimizer, scheduler,
                                    global_step, args, "best")
                    logger.info(f"  New best val reward: {best_val_reward:.4f}")

            if is_main and global_step % args.save_steps == 0:
                save_checkpoint(model, probing_heads, optimizer, scheduler,
                                global_step, args, f"step_{global_step}")

    # ── Final save ──
    if is_main:
        save_checkpoint(model, probing_heads, optimizer, scheduler,
                        global_step, args, "final")

        log_path = Path(args.output_dir) / "train_log.json"
        with open(log_path, "w") as f:
            json.dump(train_log, f, indent=2)
        logger.info(f"Training complete. Best val reward: {best_val_reward:.4f}")
        logger.info(f"Logs saved to {log_path}")

    cleanup()


def save_checkpoint(model, probing_heads, optimizer, scheduler, step, args, tag):
    save_dir = Path(args.output_dir) / f"checkpoint-{tag}"
    save_dir.mkdir(parents=True, exist_ok=True)

    inner_model = model.module if hasattr(model, 'module') else model
    inner_model.save_pretrained(save_dir / "model")

    torch.save(probing_heads.state_dict(), save_dir / "probing_heads.pt")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
    }, save_dir / "training_state.pt")

    logger.info(f"Checkpoint saved to {save_dir}")


if __name__ == "__main__":
    main()
