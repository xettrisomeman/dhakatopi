import torch

from model import DNN
from dataset import vocab
import torch.optim as optim
from torch.nn import CrossEntropyLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


UNK_IDX  = vocab['<unk>']
PAD_IDX = vocab['<pad>']
BOS_IDX = vocab['<bos>']
EOS_IDX = vocab['<eos>']
EMB_DIM = 100
HIDDEN_DIM = 50
OUT_DIM = 3
vocab_size = len(vocab)
dropout = 0.5
n_layers=2
bidirectional=True


model = DNN(
    vocab_size = vocab_size, embedding_dim = EMB_DIM, 
    hidden_dim=HIDDEN_DIM,
    output_dim = OUT_DIM,
    dropout=dropout,
    n_layers=n_layers,
    bidirectional=bidirectional,
    pad_idx=PAD_IDX
).to(device)


# don't give weight to special tokens
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMB_DIM)
model.embedding.weight.data[BOS_IDX] = torch.zeros(EMB_DIM)
model.embedding.weight.data[EOS_IDX] = torch.zeros(EMB_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMB_DIM)



optimizer = optim.Adam(model.parameters(), lr=1e-2)
# scheduler  = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-3, max_lr=1e-1)
criterion = CrossEntropyLoss().to(device)
