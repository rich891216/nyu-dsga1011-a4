import os
from transformers import T5TokenizerFast
from statistics import mean
from load_data import T5Dataset

tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
files = {}
for name in ["train.nl", "train.sql", "dev.nl", "dev.sql"]:
    with open(os.path.join("data", name), "r") as f:
        files[name] = [line.strip() for line in f]
    
# dev stats
dev_n = len(files["dev.nl"])
dev_mean_seq_len = mean([len(tokenizer.tokenize(s)) for s in files["dev.nl"]])
dev_mean_sql_len = mean([len(tokenizer.tokenize(s)) for s in files["dev.sql"]])

vocab_dev_nl = set()
for s in files["dev.nl"]:
    new_vocab = tokenizer.tokenize(s)
    vocab_dev_nl.update(new_vocab)
dev_vocab_size_nl = len(vocab_dev_nl)

vocab_dev_sql = set()
for s in files["dev.sql"]:
    new_vocab = tokenizer.tokenize(s)
    vocab_dev_sql.update(new_vocab)

dev_vocab_size_sql = len(vocab_dev_sql)

print(f"{dev_n=}, {dev_mean_seq_len=}, {dev_mean_sql_len=}, {dev_vocab_size_nl=}, {dev_vocab_size_sql=}")

# train stats
train_n = len(files["train.nl"])
train_mean_seq_len = mean([len(tokenizer.tokenize(s)) for s in files["train.nl"]])
train_mean_sql_len = mean([len(tokenizer.tokenize(s)) for s in files["train.sql"]])

vocab_train_nl = set()
for s in files["train.nl"]:
    new_vocab = tokenizer.tokenize(s)
    vocab_train_nl.update(new_vocab)
train_vocab_size_nl = len(vocab_train_nl)

vocab_train_sql = set()
for s in files["train.sql"]:
    new_vocab = tokenizer.tokenize(s)
    vocab_train_sql.update(new_vocab)

train_vocab_size_sql = len(vocab_train_sql)

print(f"{train_n=}, {train_mean_seq_len=}, {train_mean_sql_len=}, {train_vocab_size_nl=}, {train_vocab_size_sql=}")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
# -------------------------------------------- after preprocessing -------------------------------------------
train_ds = T5Dataset(DATA_DIR, "train")
dev_ds = T5Dataset(DATA_DIR, "dev")

train_nl, train_sql = train_ds.files["train.nl"], train_ds.files["train.sql"]
dev_nl, dev_sql = dev_ds.files["dev.nl"], dev_ds.files["dev.sql"]

# dev stats
dev_n = len(dev_nl)
dev_mean_seq_len = mean([len(s) for s in dev_nl])
dev_mean_sql_len = mean([len(s) for s in dev_sql])

vocab_dev_nl = set()
for s in dev_nl:
    new_vocab = s
    vocab_dev_nl.update(new_vocab)
dev_vocab_size_nl = len(vocab_dev_nl)

vocab_dev_sql = set()
for s in dev_sql:
    new_vocab = s
    vocab_dev_sql.update(new_vocab)

dev_vocab_size_sql = len(vocab_dev_sql)

print(f"{dev_n=}, {dev_mean_seq_len=}, {dev_mean_sql_len=}, {dev_vocab_size_nl=}, {dev_vocab_size_sql=}")

# train stats
train_n = len(train_nl)
train_mean_seq_len = mean([len(s) for s in train_nl])
train_mean_sql_len = mean([len(s) for s in train_sql])

vocab_train_nl = set()
for s in train_nl:
    new_vocab = s
    vocab_train_nl.update(new_vocab)
train_vocab_size_nl = len(vocab_train_nl)

vocab_train_sql = set()
for s in train_sql:
    new_vocab = s
    vocab_train_sql.update(new_vocab)

train_vocab_size_sql = len(vocab_train_sql)

print(f"{train_n=}, {train_mean_seq_len=}, {train_mean_sql_len=}, {train_vocab_size_nl=}, {train_vocab_size_sql=}")