import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, embedSize:int, hiddenSize:int, vocabSize:int,
                 linearSize:int, numClasses:int, seqLen: int, dropout: float = 0.3):
        super().__init__()

        self.em = nn.Embedding(vocabSize, embedSize)
        self.rnn = nn.LSTM(embedSize, hiddenSize, num_layers=3, batch_first=True)
        self.linear = nn.Linear(hiddenSize * seqLen, linearSize)
        self.linear2 = nn.Linear(linearSize, 1024)
        self.linear3 = nn.Linear(1024, 512)
        self.linear4 = nn.Linear(512, 128)
        self.linear5 = nn.Linear(128, 64)
        self.out = nn.Linear(64, numClasses)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Identity()

    def forward(self, x: torch.Tensor, ApplyLogit = False) -> torch.Tensor:
        #(Batch, Seq)

        y = self.em(x)
        #(Batch, Seq, Em)

        y, _ = self.rnn(y)
        #(Batch, Seq, Hidden)

        y = y.reshape((y.size(0), -1))
        #(Batch, Hidden * Seq)

        y = self.activation(self.dropout(self.linear(y)))
        #(Batch, Linear)

        y = self.activation(self.dropout(self.linear2(y)))
        #(Batch, 1024)

        y = self.activation(self.dropout(self.linear3(y)))
        # (Batch, 512)

        y = self.activation(self.dropout(self.linear4(y)))
        # (Batch, 128)

        y = self.activation(self.dropout(self.linear5(y)))
        # (Batch, 64)

        y = self.dropout(self.out(y))
        #(Batch, Classes)

        if ApplyLogit:
            y = nn.Sigmoid(y)

        return y
