from csv import reader
from data_element import DataElement
from random import randint
from pickle import dump
file = open(r"D:\MachineLearning\Datasets\AbstractClassification\train.csv", 'r', encoding="utf-8")
lines = reader(file)

categories = ["Computer Science","Physics","Mathematics",
              "Statistics","Quantitative Biology","Quantitative Finance"]


vocab = []

next(lines)

data = []

for line in lines:

    abstract = (line[1].replace(","," , ").replace(".", " . ")
                .replace("(", " ( ").replace(")", " ) ")
                .replace("{", " { ").replace("}", " } ")
                .replace("[", " [ ").replace("]", " ] ")
                .replace("/", " / ").replace("'", " ' ")
                .replace("\"", " \" ").replace("?"," ? ")
                .lower().split())

    vocab.extend(abstract)
    data.append(DataElement(abstract, [int(x) for x in line[3:]]))

print(data)

vocab.extend(["<EOS>","<SOS>","<PAD>","<UNKNOWN>"])

print(set(vocab))
file.close()

file = open("dataset.data", 'wb')
dump(data, file)
file.close()

file = open("labels.data", 'wb')
dump(categories, file)
file.close()

file = open("vocabs.data", 'wb')
dump(vocab, file)
file.close()

train = []
val = []
test = []

for x in data:
    p = randint(0, 100)

    if p <= 10:
        test.append(x)
    else:
        if p <= 30:
            val.append(x)
            if p <= 15:
               train.append(x)
        else:
            train.append(x)


file = open("train.data", 'wb')
dump(train, file)
file.close()

file = open("val.data", 'wb')
dump(val, file)
file.close()

file = open("test.data", 'wb')
dump(test, file)
file.close()

print("All : ", len(data))
print("Train : ", len(train))
print("Val : ", len(val))
print("Test : ", len(test))

print("Only Val : ", len([x for x in val if x not in train]))