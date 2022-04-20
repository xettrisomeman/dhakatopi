import torch
import torch.nn as nn


class DNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, 
    dropout, n_layers, bidirectional, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
     

    def forward(self, text, text_lengths):
        # text = [sent len, batch size]
        embed = self.dropout(self.embedding(text))
        # embed -> [sent len, batch size, embedding_dim]

        #pack sequence
        #lengths needs to be on cpu
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embed, text_lengths.to('cpu'), enforce_sorted=False)

        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # unpack sequence
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        #output -> [seq len, batch size, hid * num_direc]
        #output over padding tokens are zero tensors

        #hidden -> [num layers * num directions, batch_size, hid_dim]
        #cell = [num_layers* num_directions, batch_size, hid_dim]

        #concat tehf  inal forward (hidden[-2, :, :]) and 
        #backward (hidden(-1, :,:)) hidden layers and apply dropout

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        #hidden = [batch_size, hid_dim * num_directions]

        return self.fc(hidden)




