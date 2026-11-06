import re
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader


#*create a vocab for any text*
def make_vocab(text):
    preprocessed = re.split(r'([,.:;?_!"()\'-]|--|\s)', text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    tokens = sorted(list(set(preprocessed)))
    tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:ind for ind,token in enumerate(tokens)}
    return vocab


class SimpleTokenizer:
    def init(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for i, s in enumerate(vocab.items())}
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\'-]|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
                    item if item in self.str_to_int 
                    else "<|unk|>" for item in preprocessed
                ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    def decode(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

class gpt_dataset(Dataset):
    def init(self,txt, tokenizer, max_length, stride):
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

    #!uncomment if you want to use my tokenizer :)!
    #vocab = make_vocab(text)
    #tokenizer = SimpleTokenizer(vocab)

    dataset = gpt_dataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=shuffle, drop_last=drop_last, 
        num_workers=num_workers
    )
    return dataloader




#embedding:
#torch.nn.Embedding(vocabsize, outputdim)

def create_embeddings(tokens, vocab_size, embedding_size, max_length):
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)
    token_embeds = embedding_layer(tokens)

    pos_embedding_layer = torch.nn.Embedding(max_length, embedding_size)
    pos_embeds = pos_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeds + pos_embeds
    return input_embeddings
