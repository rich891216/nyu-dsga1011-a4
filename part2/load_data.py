import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
BOS_ID = 32099

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.files = {}
        self.split = split
        self.n = 0
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        with open(os.path.join(data_folder, split + ".nl"), "r") as f:
            s = [line.strip() for line in f]
            self.files[split + ".nl"] = []
            for i in range(len(s)):
                self.files[split + ".nl"].append(self.tokenizer.encode(s[i]))
        if split != "test":
            with open(os.path.join(data_folder, split + ".sql")) as f:
                s = [line.strip() for line in f]
                self.files[split + ".sql"] = []
                for i in range(len(s)):
                    s[i] = s[i].replace("<", "less_than")
                    s[i] = re.sub(r"'[^']*'", lambda m: m.group(0).lower(), s[i])
                    self.files[split + ".sql"].append(self.tokenizer.encode(s[i]))

        self.n = len(self.files[split + ".nl"])

    
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.split != "test":
            return self.files[self.split + ".nl"][idx], self.files[self.split + ".sql"][idx]
        else:
            return self.files[self.split + ".nl"][idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    n = len(batch)
    nl_ids = [torch.tensor(batch[i][0]) for i in range(n)]
    sq_ids = [torch.tensor([BOS_ID] + batch[i][1]) for i in range(n)]
    nl_ids = torch.nn.utils.rnn.pad_sequence(nl_ids, batch_first=True, padding_value=PAD_IDX)
    nl_mask = nl_ids != PAD_IDX

    dec_input = torch.nn.utils.rnn.pad_sequence([sq_id[:-1] for sq_id in sq_ids], batch_first=True, padding_value=PAD_IDX)
    dec_target = torch.nn.utils.rnn.pad_sequence([sq_id[1:] for sq_id in sq_ids], batch_first=True, padding_value=PAD_IDX)
    init_decoder_input = torch.tensor([BOS_ID] * n).reshape((n, 1))
    return nl_ids, nl_mask, dec_input, dec_target, init_decoder_input

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    n = len(batch)
    nl_ids = [torch.tensor(batch[i]) for i in range(n)]
    nl_ids = torch.nn.utils.rnn.pad_sequence(nl_ids, batch_first=True, padding_value=PAD_IDX)
    nl_mask = nl_ids != PAD_IDX
    init_decoder_input = torch.tensor([BOS_ID] * n).reshape((n, 1))
    return nl_ids, nl_mask, init_decoder_input

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x