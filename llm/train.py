"""Training mechanics: loss, evaluation, the LR schedule and the step loop.

Nothing in here touches the environment -- no device selection, no process
group, no file paths. main.py builds all of that and calls train_model().
"""

import os
import math
import time
import contextlib
from dataclasses import dataclass, field

import torch
import tiktoken

from tokenizer import text_to_token_ids, token_ids_to_text
from textgen import generate


@dataclass
class TrainSettings:
    """Everything tunable about a run. main.py fills in the computed fields."""

    #*a 6 GB 2060 only exposes about 5.6 GB. Measured: ~2.96 GB of params,*
    #*grads and AdamW state plus roughly 0.45 GB per sequence, and rank 0 pays*
    #*for evaluation on top of training -- micro_batch 4 OOMs there. Keep*
    #*micro_batch * grad_accum at 128 so the optimizer step still sees 65,536*
    #*tokens whatever you set these to.*
    micro_batch: int = 2         # sequences per GPU per forward pass
    grad_accum: int = 64         # micro-steps per optimizer step
    num_workers: int = 3         # DataLoader workers per rank
    seed: int = 1337

    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 500
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)   # 0.95 rather than AdamW's 0.999 default
    grad_clip: float = 1.0

    log_interval: int = 10
    eval_interval: int = 250
    eval_iters: int = 50
    ckpt_interval: int = 1000
    sample_interval: int = 2000
    out_dir: str = "checkpoints"

    #*derived in main.py once the world size and token count are known*
    max_steps: int = 0
    tokens_per_step: int = 0

    @property
    def ckpt_path(self):
        return os.path.join(self.out_dir, "ckpt.pt")


@dataclass
class Runtime:
    """Where this process is running. Built by main.setup_distributed()."""

    is_ddp: bool
    rank: int
    world_size: int
    device: torch.device
    amp_dtype: object = None

    @property
    def master(self):
        return self.rank == 0


def calc_loss_batch(input_batch, target_batch, model, rt):
    input_batch = input_batch.to(rt.device, non_blocking=True)
    target_batch = target_batch.to(rt.device, non_blocking=True)
    with torch.amp.autocast(device_type=rt.device.type, dtype=rt.amp_dtype,
                            enabled=rt.amp_dtype is not None):
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, rt, num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    num_batches = (len(data_loader) if num_batches is None
                   else min(num_batches, len(data_loader)))

    total_loss = 0.
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        total_loss += calc_loss_batch(input_batch, target_batch, model, rt).item()
    return total_loss / num_batches


def evaluate_model(model, train_eval_loader, val_loader, rt, eval_iters):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_eval_loader, model, rt, eval_iters)
        val_loss = calc_loss_loader(val_loader, model, rt, eval_iters)
    model.train()
    return train_loss, val_loss


def get_lr(step, settings):
    """Linear warmup, then cosine decay to min_lr."""
    if step < settings.warmup_steps:
        return settings.learning_rate * (step + 1) / settings.warmup_steps
    if step >= settings.max_steps:
        return settings.min_lr
    progress = ((step - settings.warmup_steps)
                / max(1, settings.max_steps - settings.warmup_steps))
    return settings.min_lr + 0.5 * (settings.learning_rate - settings.min_lr) * (
        1 + math.cos(math.pi * progress))


def endless(loader, sampler):
    """Yield batches forever, reshuffling on every pass over the data."""
    epoch = 0
    while True:
        #*without set_epoch a DistributedSampler hands out the same*
        #*permutation every epoch*
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def save_checkpoint(raw_model, optimizer, scaler, step, cfg, settings):
    torch.save({"model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "config": cfg}, settings.ckpt_path)


def sample_from(raw_model, rt, cfg, prompt="The mitochondria"):
    enc = tiktoken.get_encoding("gpt2")
    raw_model.eval()
    out = generate(raw_model, text_to_token_ids(prompt, enc).to(rt.device),
                   max_new_tokens=40, context_size=cfg["context_length"],
                   temperature=0.8, top_k=50)
    raw_model.train()
    return token_ids_to_text(out, enc)


def train_step(model, train_iter, optimizer, scaler, settings, rt):
    """One optimizer step: grad_accum micro-batches, then clip and update."""
    optimizer.zero_grad(set_to_none=True)

    accum_loss = 0.
    for micro in range(settings.grad_accum):
        input_batch, target_batch = next(train_iter)

        #*only let DDP all-reduce on the final micro-step. Without this it*
        #*ships the whole gradient across PCIe once per micro-batch, and these*
        #*cards have no NVLink to hide that behind.*
        last = micro == settings.grad_accum - 1
        sync = (contextlib.nullcontext() if (last or not rt.is_ddp)
                else model.no_sync())
        with sync:
            loss = calc_loss_batch(input_batch, target_batch, model, rt)
            scaler.scale(loss / settings.grad_accum).backward()
        accum_loss += loss.item() / settings.grad_accum

    #*unscale first so the clip sees true gradient magnitudes. With fp16 an*
    #*occasional inf here is normal -- scaler.step then skips the update.*
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
    scaler.step(optimizer)
    scaler.update()

    return accum_loss, float(grad_norm)


def train_model(model, raw_model, train_iter, optimizer, scaler,
                eval_loaders, cfg, settings, rt, start_step=0):
    """Run the training loop from start_step to settings.max_steps."""
    train_eval_loader, val_loader = eval_loaders
    train_losses, val_losses, track_tokens_seen = [], [], []

    tokens_seen = start_step * settings.tokens_per_step
    t0 = time.time()
    model.train()

    for step in range(start_step, settings.max_steps):
        lr = get_lr(step, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr

        loss, grad_norm = train_step(model, train_iter, optimizer, scaler, settings, rt)
        tokens_seen += settings.tokens_per_step

        if rt.master and step % settings.log_interval == 0:
            done = step - start_step + 1
            print(f"step {step:>6,}/{settings.max_steps:,} | loss {loss:.3f} "
                  f"| lr {lr:.2e} | gnorm {grad_norm:.2f} "
                  f"| {tokens_seen / 1e6:.1f}M tokens "
                  f"| {(time.time() - t0) / done:.2f}s/step")

        if rt.master and (step % settings.eval_interval == 0
                          or step == settings.max_steps - 1):
            train_loss, val_loss = evaluate_model(
                raw_model, train_eval_loader, val_loader, rt, settings.eval_iters)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            print(f"  eval step {step:>6,} | train {train_loss:.3f} | val {val_loss:.3f}")

        if rt.master and step > start_step and step % settings.ckpt_interval == 0:
            save_checkpoint(raw_model, optimizer, scaler, step, cfg, settings)
            print(f"  saved {settings.ckpt_path} at step {step}")

        if rt.master and step > start_step and step % settings.sample_interval == 0:
            print(f"  sample: {sample_from(raw_model, rt, cfg)!r}")

    if rt.master:
        save_checkpoint(raw_model, optimizer, scaler, settings.max_steps, cfg, settings)
        print(f"\ndone in {(time.time() - t0) / 3600:.2f}h, {tokens_seen:,} tokens seen")

    return train_losses, val_losses, track_tokens_seen
