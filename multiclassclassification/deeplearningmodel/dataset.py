import os
import pandas as pd

import torch



from nepalitokenizer import NepaliTokenizer

from sklearn.preprocessing import LabelEncoder

from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


path = os.path.join(os.path.dirname(__file__), "./newsdatanepali/")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv(path + "train.csv", usecols=['paras', 'label'])
test_data = pd.read_csv(path + "valid.csv", usecols=['paras', 'label'])


def tokenizer(text):
    tokenize = NepaliTokenizer()
    tokenized_text = tokenize.tokenizer(text)
    return tokenized_text


encoder = LabelEncoder()
encoder.fit(train_data.label.value_counts().index)

train_data['tokens'] = train_data['paras'].apply(tokenizer)
train_data['label'] = encoder.transform(train_data['label'])


test_data['tokens'] = test_data['paras'].apply(tokenizer)
test_data['label'] = encoder.transform(test_data['label'])


# change to lists
def change_to_lists(data, length):
    datas = []
    for idx in range(length):
        d = data.iloc[idx]
        datas.append((d.tokens, d.label))
    return datas

train_data_list = change_to_lists(train_data, train_data.shape[0])
test_data_list = change_to_lists(test_data, test_data.shape[0])


# create a vocab
def yield_tokens(data):
    for text, _ in data:
        yield text

vocab = build_vocab_from_iterator(yield_tokens(train_data_list), specials=['<unk>', '<eos>', '<bos>', '<pad>'], min_freq=1, )
vocab.set_default_index(vocab['<unk>'])


text_transform = lambda x: [vocab['<bos>']] + [vocab[token] for token in x] + [vocab['<eos>']]
# print(text_transform(train_data.loc[1].tokens))


def collate_batch(batch):
    label_list, text_list, text_lengths = [], [], []
    # sort the batch
    sorted_batch = sorted(batch, key=lambda x: x[0], reverse=False)
    for _text, _label in sorted_batch:
        # transform tokens to vocab
        text = torch.tensor(text_transform(_text), dtype=torch.int64)
        text_list.append(text)
        label_list.append(_label)
        text_lengths.append(text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return label_list.to(device), pad_sequence(text_list, padding_value=3.0).to(device), torch.tensor(text_lengths, dtype=torch.int64)
    

train_dataloader = DataLoader(train_data_list, batch_size=16, collate_fn=collate_batch, shuffle=False)
test_dataloader = DataLoader(test_data_list, batch_size=16, collate_fn=collate_batch, shuffle=False)

