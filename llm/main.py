"""Entry point for pretraining on the token files prepare_data.py produced.

    python main.py                                # single GPU / CPU
    torchrun --nproc_per_node=2 main.py           # both 2060s

Everything environment-shaped lives here: process group, device, loaders,
model, optimizer. The loop itself is in train.py.
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import configs
import tokenizer
import train
import model as model_file
from train import TrainSettings, Runtime


DATA_DIR = "data"
RESUME = True        # continue from checkpoints/ckpt.pt when it exists


def setup_distributed():
    """Join the process group under torchrun, otherwise run solo."""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        #*bind this process to its GPU before anything is allocated on it*
        torch.cuda.set_device(local_rank)
        return Runtime(is_ddp=True, rank=dist.get_rank(),
                       world_size=dist.get_world_size(),
                       device=torch.device("cuda", local_rank))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        major, minor = map(int, torch.__version__.split(".")[:2])
        device = torch.device("mps") if (major, minor) >= (2, 9) else torch.device("cpu")
    else:
        device = torch.device("cpu")
    return Runtime(is_ddp=False, rank=0, world_size=1, device=device)


def pick_amp_dtype(device):
    #fp16 on Turing, bf16 on Ampere and newer
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def build_loaders(cfg, settings, rt):
    """Training loader (sharded across ranks) plus rank-0 evaluation loaders."""
    ctx = cfg["context_length"]
    train_bin = os.path.join(DATA_DIR, "train.bin")

    if rt.is_ddp:
        train_loader, sampler = tokenizer.make_bin_loader(
            train_bin, batch_size=settings.micro_batch, context_length=ctx,
            stride=ctx, shuffle=True, drop_last=True,
            num_workers=settings.num_workers, distributed=True)
    else:
        train_loader = tokenizer.make_bin_loader(
            train_bin, batch_size=settings.micro_batch, context_length=ctx,
            stride=ctx, shuffle=True, drop_last=True,
            num_workers=settings.num_workers)
        sampler = None

    #*evaluation runs on rank 0 only, so these stay non-distributed. Both are*
    #*unshuffled so the numbers are comparable from one eval to the next --*
    #*measuring train loss on the shuffled loader would move around on its own.*
    train_eval_loader = val_loader = None
    if rt.master:
        train_eval_loader = tokenizer.make_bin_loader(
            train_bin, batch_size=settings.micro_batch, context_length=ctx,
            stride=ctx, shuffle=False, drop_last=False, num_workers=1)
        val_loader = tokenizer.make_bin_loader(
            os.path.join(DATA_DIR, "val.bin"), batch_size=settings.micro_batch,
            context_length=ctx, stride=ctx, shuffle=False, drop_last=False,
            num_workers=1)

    return train_loader, sampler, (train_eval_loader, val_loader)


def build_model(cfg, rt):
    """Returns (model_for_training, unwrapped_model)."""
    model = model_file.GPTModel(cfg)
    model.to(rt.device)
    if rt.is_ddp:
        #*DDP broadcasts rank 0's weights at construction, so the per-rank*
        #*seed does not leave the replicas out of step*
        return DDP(model, device_ids=[rt.device.index]), model
    return model, model


def maybe_resume(raw_model, optimizer, scaler, settings, rt):
    if not (RESUME and os.path.exists(settings.ckpt_path)):
        return 0
    ckpt = torch.load(settings.ckpt_path, map_location=rt.device, weights_only=False)
    raw_model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    if rt.master:
        print(f"resumed from {settings.ckpt_path} at step {ckpt['step'] + 1}")
    return ckpt["step"] + 1


def print_banner(cfg, settings, rt, raw_model, meta):
    n_params = sum(p.numel() for p in raw_model.parameters())
    tied = raw_model.out_head.weight is raw_model.tok_emb.weight
    print(f"device      {rt.device} | ddp={rt.is_ddp} world={rt.world_size} "
          f"| amp={rt.amp_dtype}")
    print(f"data        {meta['train_tokens']:,} train / {meta['val_tokens']:,} val tokens")
    print(f"model       {n_params / 1e6:.1f}M params | weights tied: {tied}")
    if not tied:
        print("            (tying out_head to tok_emb frees ~0.6 GB per GPU)")
    print(f"batch       {settings.tokens_per_step:,} tokens/step "
          f"({rt.world_size} x {settings.micro_batch} x {settings.grad_accum} "
          f"x {cfg['context_length']})")
    print(f"schedule    {settings.max_steps:,} steps | {settings.warmup_steps} warmup "
          f"| lr {settings.learning_rate} -> {settings.min_lr}\n")


def main():
    rt = setup_distributed()
    rt.amp_dtype = pick_amp_dtype(rt.device)

    cfg = configs.GPT_CONFIG_124M
    settings = TrainSettings()
    torch.manual_seed(settings.seed + rt.rank)

    meta = tokenizer.load_meta(DATA_DIR)
    assert meta["max_token_id"] < cfg["vocab_size"], (
        f"data holds token id {meta['max_token_id']}, "
        f"model vocab is {cfg['vocab_size']}")

    settings.tokens_per_step = (rt.world_size * settings.micro_batch
                                * settings.grad_accum * cfg["context_length"])
    settings.max_steps = meta["train_tokens"] // settings.tokens_per_step

    train_loader, sampler, eval_loaders = build_loaders(cfg, settings, rt)
    model, raw_model = build_model(cfg, rt)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate,
                                  betas=settings.betas,
                                  weight_decay=settings.weight_decay)
    #*the scaler is only needed for fp16; under bf16 it stays disabled and*
    #*every scaler call becomes a no-op*
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(rt.amp_dtype == torch.float16))

    if rt.master:
        os.makedirs(settings.out_dir, exist_ok=True)
    start_step = maybe_resume(raw_model, optimizer, scaler, settings, rt)

    if rt.master:
        print_banner(cfg, settings, rt, raw_model, meta)

    train.train_model(
        model, raw_model, train.endless(train_loader, sampler),
        optimizer, scaler, eval_loaders, cfg, settings, rt, start_step)

    if rt.is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
