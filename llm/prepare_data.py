"""
Turn a FineWeb-Edu parquet shard into flat uint16 token files for pretraining.

Run once, offline:
    python prepare_data.py --parquet data/000_00000.parquet --out data

Writes train.bin, val.bin and meta.json into --out. The .bin files are raw
uint16 arrays of GPT-2 token ids with <|endoftext|> between documents; train.py
memory-maps them instead of holding tokens in RAM.

Get the shard first (~2.15 GB, about 750M tokens):
    curl -L -o data/000_00000.parquet \
      https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet
"""

import argparse
import json
import os
import random
import time
import zlib

import numpy as np
import pyarrow.parquet as pq
import tiktoken


#*FineWeb-Edu has already had trafilatura extraction, fastText language ID,*
#*Gopher/C4 quality filters, MinHash dedup and the educational-quality*
#*classifier applied upstream. Re-running any of that costs tokens and buys*
#*nothing, so we only filter on metadata columns (free) plus one check that is*
#*specific to our tokenizer and could not have been done upstream.*

MIN_LANGUAGE_SCORE = 0.80    # upstream default is 0.65
MIN_INT_SCORE = 3            # educational quality; upstream already keeps >= 3
MAX_TOKENS_PER_BYTE = 0.5    # English prose sits around 0.25-0.30

VAL_FRACTION = 0.01
DOC_BUFFER = 8192            # documents tokenized per batch
EOT = 50256                  # <|endoftext|>, our document separator
DTYPE = np.uint16            # valid because the GPT-2 vocab is 50257 < 65536


class BinWriter:
    """Append-only writer for a flat uint16 token file."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "wb")
        self.n_tokens = 0
        self.n_docs = 0
        self.max_id = 0

    def write(self, ids):
        #*numpy raises if any id is out of range for uint16, so a corrupt*
        #*encode fails loudly here rather than silently wrapping around*
        arr = np.asarray(ids, dtype=DTYPE)
        arr.tofile(self.f)
        self.n_tokens += arr.size
        self.n_docs += 1
        self.max_id = max(self.max_id, int(arr.max()))

    def close(self):
        self.f.close()


def is_val(doc_id):
    #*split on a stable hash of the document id, not Python's hash(), which is*
    #*randomized per process. Document-level so no document can straddle the*
    #*train/val boundary.*
    return zlib.crc32(doc_id.encode("utf-8")) % 1000 < int(VAL_FRACTION * 1000)


def iter_documents(parquet_path, stats, limit=None):
    """Stream the shard, yielding (id, text, reported_tokens) past metadata filters."""
    columns = ["id", "text", "language_score", "int_score", "token_count"]
    pf = pq.ParquetFile(parquet_path)

    for batch in pf.iter_batches(batch_size=DOC_BUFFER, columns=columns):
        d = batch.to_pydict()
        for i in range(len(d["id"])):
            if limit is not None and stats["docs_seen"] >= limit:
                return
            stats["docs_seen"] += 1

            if d["language_score"][i] < MIN_LANGUAGE_SCORE:
                stats["dropped_language"] += 1
                continue
            if d["int_score"][i] < MIN_INT_SCORE:
                stats["dropped_score"] += 1
                continue

            yield d["id"][i], d["text"][i], d["token_count"][i]


def iter_buffers(parquet_path, stats, rng, limit=None):
    """Group documents into shuffled buffers so one long document's samples are
    not written as a long contiguous run."""
    buf = []
    for doc in iter_documents(parquet_path, stats, limit):
        buf.append(doc)
        if len(buf) >= DOC_BUFFER:
            rng.shuffle(buf)
            yield buf
            buf = []
    if buf:
        rng.shuffle(buf)
        yield buf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True, help="path to a FineWeb-Edu parquet shard")
    p.add_argument("--out", default="data", help="output directory")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="tiktoken encode threads")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N documents (smoke test)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--force", action="store_true", help="overwrite existing .bin files")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    train_path = os.path.join(args.out, "train.bin")
    val_path = os.path.join(args.out, "val.bin")
    if not args.force and (os.path.exists(train_path) or os.path.exists(val_path)):
        raise SystemExit(f"{train_path} or {val_path} already exists; pass --force to overwrite")

    enc = tiktoken.get_encoding("gpt2")
    rng = random.Random(args.seed)

    stats = {
        "docs_seen": 0,
        "dropped_language": 0,
        "dropped_score": 0,
        "dropped_tokens_per_byte": 0,
        "dropped_empty": 0,
        "total_bytes": 0,
        "total_chars": 0,
        "reported_tokens": 0,   # from the token_count column, for cross-checking
    }

    train = BinWriter(train_path)
    val = BinWriter(val_path)
    start = time.time()

    for buf in iter_buffers(args.parquet, stats, rng, args.limit):
        #*tiktoken releases the GIL and threads internally, so this gets us*
        #*parallelism without shipping gigabytes of text between processes*
        encoded = enc.encode_ordinary_batch([t for _, t, _ in buf], num_threads=args.workers)

        for (doc_id, text, reported), ids in zip(buf, encoded):
            nbytes = len(text.encode("utf-8"))
            if nbytes == 0 or not ids:
                stats["dropped_empty"] += 1
                continue

            #*the one filter FineWeb could not apply for us: GPT-2 BPE falls*
            #*back to raw bytes for non-Latin text, so anything that slipped*
            #*past language ID costs several times the tokens per character*
            if len(ids) / nbytes > MAX_TOKENS_PER_BYTE:
                stats["dropped_tokens_per_byte"] += 1
                continue

            stats["total_bytes"] += nbytes
            stats["total_chars"] += len(text)
            stats["reported_tokens"] += reported

            ids.append(EOT)
            (val if is_val(doc_id) else train).write(ids)

        elapsed = time.time() - start
        written = train.n_tokens + val.n_tokens
        print(f"{stats['docs_seen']:>9,} docs seen | {written:>12,} tokens | "
              f"{elapsed:6.0f}s | {stats['total_bytes'] / 1e6 / max(elapsed, 1e-9):5.1f} MB/s")

    train.close()
    val.close()
    elapsed = time.time() - start

    kept = train.n_docs + val.n_docs
    actual_tokens = train.n_tokens + val.n_tokens - kept  # minus the EOT we appended
    chars_per_token = stats["total_chars"] / max(actual_tokens, 1)

    meta = {
        "tokenizer": "gpt2",
        "dtype": "uint16",
        "eot_id": EOT,
        "train_tokens": train.n_tokens,
        "val_tokens": val.n_tokens,
        "train_docs": train.n_docs,
        "val_docs": val.n_docs,
        "max_token_id": max(train.max_id, val.max_id),
        "chars_per_token": round(chars_per_token, 3),
        "filters": {
            "min_language_score": MIN_LANGUAGE_SCORE,
            "min_int_score": MIN_INT_SCORE,
            "max_tokens_per_byte": MAX_TOKENS_PER_BYTE,
            "val_fraction": VAL_FRACTION,
        },
        "source": {
            "path": os.path.abspath(args.parquet),
            "bytes": os.path.getsize(args.parquet),
        },
        "stats": stats,
        "seconds": round(elapsed, 1),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    #*sanity checks -- cheaper to do here than to debug mid-training*
    print(f"\ndone in {elapsed / 60:.1f} min")
    print(f"  train      {train.n_tokens:>12,} tokens  /  {train.n_docs:>9,} docs")
    print(f"  val        {val.n_tokens:>12,} tokens  /  {val.n_docs:>9,} docs")
    print(f"  on disk    {(train.n_tokens + val.n_tokens) * 2 / 1e9:.2f} GB")

    print("\ndrops:")
    for key in ("dropped_language", "dropped_score", "dropped_tokens_per_byte", "dropped_empty"):
        n = stats[key]
        print(f"  {key:<26} {n:>9,}  ({100 * n / max(stats['docs_seen'], 1):5.2f}%)")
    print("  (if any single rule removes a large share, suspect the rule before the data)")

    print("\nchecks:")
    max_id = max(train.max_id, val.max_id)
    print(f"  max token id               {max_id:,}  (must be < vocab_size)")
    print(f"  chars per token            {chars_per_token:.2f}  (healthy English: 3.5-4.0)")
    if stats["reported_tokens"]:
        drift = 100 * (actual_tokens - stats["reported_tokens"]) / stats["reported_tokens"]
        print(f"  vs token_count column      {drift:+.2f}%  (large drift means a bad encode path)")
    for name, w in (("train", train), ("val", val)):
        size = os.path.getsize(w.path)
        ok = "ok" if size == w.n_tokens * 2 else f"MISMATCH (expected {w.n_tokens * 2})"
        print(f"  {name}.bin size              {size:,} bytes  {ok}")

    #*read a sample back through the memmap path train.py will use, and decode*
    #*it, so a silent dtype or offset bug shows up as garbage right now*
    if train.n_tokens > 512:
        data = np.memmap(train_path, dtype=DTYPE, mode="r")
        off = rng.randrange(0, len(data) - 256)
        print(f"\nsample decoded from train.bin at offset {off:,}:\n")
        print(enc.decode(data[off:off + 256].tolist()))


if __name__ == "__main__":
    main()
