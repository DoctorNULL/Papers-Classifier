import math

import torch.cuda

from Vocabs import Vocab
from datahandler import DataHandler
from torch.utils.data import DataLoader
from model import Model
import torch.nn as nn



batchSize = 4000
embedSize = 256
hiddenSize = 256
linearSize = 2048
classesSize = 6
maxLen = 50

weight = 185

epoches = 200
lr = 0.0005
dev = "cuda" if torch.cuda.is_available() else "cpu"

vocab = Vocab(maxLen=maxLen)

train = DataHandler(vocab)
val = DataHandler(vocab, "val")
trainLoader = DataLoader(train, batchSize, True)
valLoader = DataLoader(val, batchSize, True)

#model = Model(embedSize, hiddenSize, len(vocab), linearSize, classesSize, maxLen).to(dev)
model = torch.load("model.pt").to(dev)
optim = torch.optim.Adam(model.parameters(),lr)

bestAccuracy = math.inf

print("Start Training...")

for epoch in range(1, epoches + 1):

    model.train()

    trainCorrect = 0
    trainCost = 0
    totalCorrectOnes = 0
    totalOnes = 0

    for idx, (x, y) in enumerate(trainLoader):

        predicted = model(x.to(dev))
        y_ = predicted.sigmoid().round()
        y = y.to(dev)
        w = y * weight + (1 - y) * (200 - weight)
        cost = nn.BCEWithLogitsLoss(w, reduction="sum")
        loss = cost(predicted, y)
        correct = (y_ == y).sum().sum().item()
        correctOnes = ((y_ == y) & (y_ == 1)).sum().sum().item()
        trainCost += loss.sum().sum()
        trainCorrect += correct

        totalCorrectOnes += correctOnes
        totalOnes += y.sum().sum().sum().item()

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Epoch {epoch}/{epoches} Training Batch {idx * batchSize}/{len(train)} Loss {loss.sum().sum().item()}"
              f" Accuracy {correct / (x.size(0) * classesSize) * 100}"
              f" Correct {correct}/{x.size(0) * classesSize}"
              f" Correct Ones {correctOnes}/{y.sum().sum().sum().item()}"
              f" Ones Accuracy {correctOnes/y.sum().sum().sum().item() * 100}")

    model.eval()

    valCorrect = 0
    valCost = 0

    with torch.no_grad():
        for idx, (x, y) in enumerate(valLoader):
            predicted = model(x.to(dev))
            y_ = predicted.sigmoid().round()
            #predicted = torch.where(predicted >= confidence, predicted.ceil(), predicted.floor())
            y = y.to(dev)
            w = y * weight + (1 - y) * (200 - weight)
            cost = nn.BCEWithLogitsLoss(w, reduction="sum")
            loss = cost(predicted, y)
            correct = (y_ == y).sum().sum().item()
            correctOnes = ((y_ == y) & (y_ == 1)).sum().sum().item()

            valCost += loss
            valCorrect += correct

            print(f"Epoch {epoch}/{epoches} Evaluation Batch {idx * batchSize}/{len(val)} Loss {loss}"
                  f" Accuracy {correct / (x.size(0) * classesSize) * 100}"
                  f" Correct {correct}/{x.size(0) * classesSize}"
                  f" Correct Ones {correctOnes}/{y.sum().sum().sum().item()}"
                  f" Ones Accuracy {correctOnes / y.sum().sum().sum().item() * 100}")

    print("-"*50)
    print(f"Epoch {epoch}/{epoches} Training Cost {trainCost}"
          f" Training Accuracy {trainCorrect / (len(train) * classesSize) * 100} "
          f" Training Correct {trainCorrect}/{len(train) * classesSize}")
    print(f"Training Correct Ones {totalCorrectOnes}/{totalOnes}"
          f" Training Ones Accuracy {totalCorrectOnes/totalOnes * 100}")
    print(f"Validation Cost {valCost}"
          f" Validation Accuracy {valCorrect / (len(val) * classesSize) * 100} "
          f" Validation Correct {valCorrect}/{len(val) * classesSize}")
    print("-" * 50)

    torch.save(model.to("cpu"), "model.pt")

    if valCorrect / (len(val) * classesSize) * 100 > bestAccuracy:
        torch.save(model.to("cpu"), "bestModel.pt")

    model.to(dev)
