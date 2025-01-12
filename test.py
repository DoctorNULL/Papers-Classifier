import torch.cuda

from Vocabs import Vocab
from datahandler import DataHandler
from torch.utils.data import DataLoader
from model import Model
import torch.nn as nn



batchSize = 500
maxLen = 50
classesSize = 6
confidence = 0.9
weight = 10

dev = "cuda" if torch.cuda.is_available() else "cpu"

vocab = Vocab(maxLen=maxLen)

test = DataHandler(vocab, "test")
testLoader = DataLoader(test, batchSize, True)

model = torch.load("model.pt").to(dev)

print("Start Testing...")

model.eval()

testCorrect = 0
testCost = 0
totalCorrectOnes = 0
totalOnes = 0

with torch.no_grad():
    for idx, (x, y) in enumerate(testLoader):
        predicted = model(x.to(dev))
        y_ = predicted.sigmoid().round()
        y = y.to(dev)
        w = y * weight + (1 - y) * (100 - weight)
        cost = nn.BCEWithLogitsLoss(w)
        loss = cost(predicted, y)
        correct = ((torch.sigmoid(predicted).round()) == y).sum().sum().item()
        correctOnes = ((y_ == y) & (y_ == 1)).sum().sum().item()

        totalCorrectOnes += correctOnes
        totalOnes += y.sum().sum().sum().item()

        testCost += loss
        testCorrect += correct

        print(f"Testing Batch {idx * batchSize}/{len(test)} Loss {loss}"
              f" Accuracy {correct / (batchSize * classesSize) * 100} Correct {correct}/{batchSize * classesSize}"
              f" Correct Ones {correctOnes}/{y.sum().sum().sum().item()}"
              f" Ones Accuracy {correctOnes / y.sum().sum().sum().item() * 100}")

print(f"Testing Cost {testCost}"
      f" Testing Accuracy {testCorrect / (len(test) * classesSize) * 100} "
      f" Testing Correct {testCorrect}/{len(test) * classesSize}")
