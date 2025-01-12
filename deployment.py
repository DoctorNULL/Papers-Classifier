import torch.cuda

from Vocabs import Vocab
from datahandler import DataHandler
from model import Model
import torch.nn as nn
from pickle import load


maxLen = 50
classesSize = 6

dev = "cuda" if torch.cuda.is_available() else "cpu"

vocab = Vocab(maxLen=maxLen)

test = DataHandler(vocab, "test")

model = torch.load("model.pt").to(dev)
cost = nn.BCEWithLogitsLoss()

file = open("labels.data", 'rb')
labels = list(load(file))
file.close()

print(labels)

print("Start Deploying...")

model.eval()

with torch.no_grad():
    inp = None
    y = None
    while inp != "exit":
        inp = input("Enter Number or Caption : ")
        try:
            inp = int(inp)

            print("Abstract : ", " ".join(test.data[inp].abstract))

            gp = []

            for i,ele in enumerate(test.data[inp].labels):
                if bool(int(ele)):
                    gp.append(labels[i])

            print("Ground Truth : ", gp)

            x,y = test[inp]

            predicted = model(x.unsqueeze(0).to(dev))
            y = y.unsqueeze(0).to(dev)
            loss = cost(predicted, y)
            predicted = torch.sigmoid(predicted).round()

            correct = (predicted == y).sum().sum().item()

            mask = (predicted == 1).squeeze().nonzero(as_tuple=True)[0]

            res = []

            for i in mask:
                res.append(labels[i.item()])

            print(res)

            print(f"Deployment Batch Loss {loss}"
                  f" Accuracy {correct / classesSize * 100} Correct {correct}/{classesSize}")

        except Exception as e:
            print("Abstract : ", inp)

            x = vocab.encode(inp.replace(","," , ").replace(".", " . ")
                .replace("(", " ( ").replace(")", " ) ")
                .replace("{", " { ").replace("}", " } ")
                .replace("[", " [ ").replace("]", " ] ")
                .replace("/", " / ").replace("'", " ' ")
                .replace("\"", " \" ").replace("?"," ? ")
                .lower().split())

            x = torch.tensor(x).unsqueeze(0).to(dev)

            predicted = model(x)
            predicted = torch.sigmoid(predicted).round()

            mask = (predicted == 1).squeeze().nonzero(as_tuple=True)[0]

            res = []

            for i in mask:
                res.append(labels[i.item()])

            print(res)

