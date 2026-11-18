"""Load a training checkpoint and generate text from it.

    python sample.py --prompt "The mitochondria"
    python sample.py --prompt "Once upon a time" -n 3 --temperature 0.9 --top-k 50
    python sample.py -i                           # interactive prompt loop

"""

import os
import argparse

import torch
import tiktoken

import model as model_file
from tokenizer import text_to_token_ids, token_ids_to_text
from textgen import generate


DEFAULT_CKPT = os.path.join("checkpoints", "ckpt.pt")
EOT_ID = 50256   # tiktoken gpt2's <|endoftext|>


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path, device):
    """Rebuild the model from the config saved next to the weights.

    Reading the config out of the checkpoint rather than configs.py means an
    old checkpoint still loads after GPT_CONFIG_124M has been edited.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = model_file.GPTModel(cfg)

    #*train.py saves raw_model.state_dict(), so keys are unprefixed. A*
    #*state_dict taken from the DDP wrapper instead would carry "module." --*
    #*strip it so either file loads.*
    state = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)

    model.to(device).eval()
    return model, cfg, ckpt.get("step")


def sample(model, cfg, prompt, enc, device, max_new_tokens, temperature,
           top_k, stop_at_eot):
    #*the model only has pos_emb and a causal mask for context_length*
    #*positions, so a prompt longer than that cannot be fed in whole --*
    ids = text_to_token_ids(prompt, enc).to(device)
    if ids.shape[1] > cfg["context_length"]:
        ids = ids[:, -cfg["context_length"]:]

    out = generate(model, ids, max_new_tokens=max_new_tokens,
                   context_size=cfg["context_length"],
                   temperature=temperature, top_k=top_k,
                   eos_id=EOT_ID if stop_at_eot else None)
    return token_ids_to_text(out, enc)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint path")
    p.add_argument("--prompt", default="The mitochondria")
    p.add_argument("-n", "--num-samples", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8,
                   help="0 is greedy/deterministic, higher is more random")
    p.add_argument("--top-k", type=int, default=50,
                   help="0 disables top-k filtering")
    p.add_argument("--stop-at-eot", action="store_true",
                   help="stop a sample when <|endoftext|> is drawn")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cuda / cpu / mps")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="keep asking for prompts until EOF or ctrl-c")
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = pick_device(args.device)
    model, cfg, step = load_model(args.ckpt, device)
    enc = tiktoken.get_encoding("gpt2")

    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"{args.ckpt} | step {step:,} | {n_params / 1e6:.1f}M params "
          f"| ctx {cfg['context_length']} | {device}\n")

    top_k = args.top_k if args.top_k > 0 else None

    def run(prompt):
        for i in range(args.num_samples):
            text = sample(model, cfg, prompt, enc, device, args.max_new_tokens,
                          args.temperature, top_k, args.stop_at_eot)
            if args.num_samples > 1:
                print(f"--- sample {i + 1} ---")
            print(text)
            print()

    if not args.interactive:
        run(args.prompt)
        return

    while True:
        try:
            prompt = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if prompt.strip():
            run(prompt)


if __name__ == "__main__":
    main()
