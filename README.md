## Overview

Decoder only LLM based on transformer architecture, utilizing the default GPT-2 124M parameter config (a shorter context length is used to shorten training time). PyTorch based and Optimized to utilize multiple GPUs during training.

This repository in part serves to document learning progress as I read Sebastian Raschka's "Build a Large Language Model from Scratch" book, and my chapter notes are located in the "chapter_notes" folder. All core LLM code is located in the "llm" folder.

## Prerequisites
The following packages are needed:
- PyTorch
- NumPy
- Pandas
- PyYaml
- PyArrow
- Tiktoken
- NVIDIA CUDA GPUs (if multi-gpu training)

If you have FlakeOS, the `flake.nix` and `flake.lock` files are located in this repo for instant use.


## Getting Started
After forking/cloning the repo, you can either train the model on your very own machine, or immediately generate text samples.

### Training the model
If you'd like to use the same  ~750M parameter FineWeb-Edu shard as I did, download the .parquet file [here](https://huggingface.co/oskarexists/128m_gpt_scratch_weights/tree/main). Otherwise, feel free to use whatever other data shard you want to use (in .parquet format). Then, build your training and validation sets (.bin files) using the following command:

```
python prepare_data.py --parquet data/000_00000.parquet --out data
```
Replacing `data/000_00000.parquet` with wherever your .parquet file is stored, and `data` with wherever you want your train/validation sets to go (will create new directory if it doesn't already exist). Note that the directory will grow to about 3.6GB, and If you do decide to store your sets in a directoy named other than `data/`, make sure to change `DATA_DIR` in `main.py` so the loop knows where your data is.

To begin training, run one of the following:
```
    python main.py                          # single GPU / CPU
    torchrun --nproc_per_node=n main.py     # n GPUs (replace n)
```
Checkpoints will be automatically saved every 1000 train steps to to a `checkpoints` directory (roughly 2GB). It took me around 15 hours to train on two RTX 2060 6GBs.

### Text generation
After obtaining weights either by training or utilizing pretrained weights (mine are located [here](https://huggingface.co/oskarexists/128m_gpt_scratch_weights/tree/main)), you can test text generation by running one of the following commands:

```
    python sample.py --prompt "prompt here"
    python sample.py -i                           # interactive prompt loop
```

Additional flags and their effects can be found in `textgen.py` for enhanced text viewing. Note that the model is NOT (yet) fine tuned for querying, and the dataset it was trained on isn't very big, so responses may be somewhat nonsensical, albeit cohesive.