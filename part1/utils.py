import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    QWERTY_NEIGHBORS = {
        "q": "wa",     "w": "qeas",   "e": "wrsd",   "r": "etdf",   "t": "ryfg",
        "y": "tugh",   "u": "yihj",   "i": "uojk",   "o": "ipkl",   "p": "ol",
        "a": "sqwz",   "s": "adwezx", "d": "sferxc", "f": "dgrtcv", "g": "fhtyvb",
        "h": "gjyubn", "j": "hkuinm", "k": "jliom",  "l": "kop",
        "z": "xas",    "x": "zcsd",   "c": "xvdf",   "v": "cbfg",   "b": "vngh",
        "n": "bmhj",   "m": "njk",
    }
    text = example["text"].lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text_list = list(text)
    valid_idxs = []
    for i in range(len(text_list)):
        if text_list[i].isascii() and text_list[i].isalpha():
            valid_idxs.append(i)

    k = round(0.05 * len(valid_idxs))
    chosen = random.sample(valid_idxs, k)
    for idx in chosen:
        rand_typo = random.randint(0, len(QWERTY_NEIGHBORS[text_list[idx]])-1)
        text_list[idx] = QWERTY_NEIGHBORS[text_list[idx]][rand_typo]
    example["text"] = "".join(text_list)
    ##### YOUR CODE ENDS HERE ######

    return example
