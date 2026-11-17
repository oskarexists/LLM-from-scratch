import os
import re
import json
import torch
import numpy as np
import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP




#*create a vocab for any text*
def make_vocab(text):
    preprocessed = re.split(r'([,.:;?_!"()\'-]|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    tokens = sorted(list(set(preprocessed)))
    tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:ind for ind,token in enumerate(tokens)}
    return vocab


class gpt_dataset(Dataset):
    def __init__(self,txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #*ensure there are enough token to process into a dataset*
        token_ids = tokenizer.encode(txt)
        assert len(token_ids) > max_length, "# of tokens must be at least max_length + 1"

        #*chunk the inputs with sliding window of max_length size*
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:1+i+max_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


def make_data_loader(text, batch_size = 4, max_length = 256, 
                     stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    
    #*choose which tokenizer to use, mine (SimpleTokenizer) or tiktoken BPE*
    #*defaulting to tiktoken BPE for better usability*
    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = gpt_dataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=shuffle, drop_last=drop_last, 
        num_workers=num_workers
    )
    return dataloader


class BinDataset(Dataset):
    """Reads training samples straight out of a .bin written by prepare_data.py.

    Nothing is materialised up front: the file is memory-mapped and each sample
    is a single slice, so a 1.5 GB token file costs essentially no RAM. Each
    slice is context_length + 1 tokens long because the targets are just the
    inputs shifted one position to the left.
    """

    def __init__(self, bin_path, context_length, stride=None):
        self.path = bin_path
        self.ctx = context_length
        self.stride = stride or context_length
        self.n_tokens = os.path.getsize(bin_path) // np.dtype(np.uint16).itemsize

        assert self.n_tokens > self.ctx, (
            f"{bin_path} holds {self.n_tokens} tokens, need more than {self.ctx}")

        #*opened lazily in __getitem__ so that each DataLoader worker builds its*
        #*own mapping -- a memmap created here would be inherited through fork*
        self.data = None

    def __len__(self):
        #*the last window must satisfy  i*stride + ctx + 1 <= n_tokens, so that*
        #*there is always one extra token available for the shifted target*
        return (self.n_tokens - self.ctx - 1) // self.stride + 1

    def __getitem__(self, index):
        if self.data is None:
            self.data = np.memmap(self.path, dtype=np.uint16, mode="r")

        start = index * self.stride
        chunk = self.data[start : start + self.ctx + 1]

        #*astype copies the slice out of the mapping and widens it to int64,*
        #*which is what nn.Embedding and cross_entropy both expect*
        inputs = torch.from_numpy(chunk[:-1].astype(np.int64))
        targets = torch.from_numpy(chunk[1:].astype(np.int64))
        return inputs, targets


def make_bin_loader(bin_path, batch_size=8, context_length=256, stride=None,
                    shuffle=True, drop_last=True, num_workers=4, pin_memory=True,
                    distributed=False):
    """Build a DataLoader over a .bin token file.

    Returns the DataLoader on its own, or a (DataLoader, DistributedSampler)
    pair when distributed=True -- the training loop needs the sampler so it can
    call set_epoch() and get a different shuffle each pass.
    """
    dataset = BinDataset(bin_path, context_length, stride)

    sampler = None
    if distributed:
        assert dist.is_available() and dist.is_initialized(), (
            "distributed=True needs an initialised process group; call "
            "init_process_group() first and launch with torchrun")

        #*each rank must see a disjoint slice of the data. Leaving shuffle=True*
        #*on the DataLoader instead would have every rank draw from the whole*
        #*file, so the second GPU would add compute without adding progress.*
        #*drop_last=False here makes the sampler pad by repeating samples so*
        #*every rank gets an equal count, which skews a validation average --*
        #*prefer distributed=False and a rank-0-only loader for val.*
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        #*sampler and shuffle are mutually exclusive; passing both raises*
        sampler=sampler,
        shuffle=False if sampler is not None else shuffle,
        drop_last=drop_last,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )

    if distributed:
        return dataloader, sampler
    return dataloader


def load_meta(data_dir):
    """Read the meta.json prepare_data.py wrote and check it matches this code."""
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    assert meta["dtype"] == "uint16", f"unexpected dtype {meta['dtype']}"
    assert meta["tokenizer"] == "gpt2", f"unexpected tokenizer {meta['tokenizer']}"
    return meta


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

#embedding:
#torch.nn.Embedding(vocabsize, outputdim)

def create_embeddings(tokens, vocab_size, embedding_size, max_length):
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
    token_embeds = embedding_layer(tokens)

    pos_embedding_layer = torch.nn.Embedding(max_length, embedding_size)
    pos_embeds = pos_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeds + pos_embeds
    return input_embeddings
