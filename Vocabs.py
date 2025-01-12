from pickle import load


class Vocab(object):
    def __init__(self, path=r"vocabs.data", maxLen = 400):

        file = open(path, "rb")
        self.vocab = list(load(file))
        file.close()


        self.maxLen = maxLen
        self.sos = self.vocab.index("<SOS>")
        self.eos = self.vocab.index("<EOS>")
        self.pad = self.vocab.index("<PAD>")
        self.unknown = self.vocab.index("<UNKNOWN>")

    def encode(self, data: list) -> list:
        res = [self.sos]

        for x in data:
            try:
                res.append(self.vocab.index(x))
            except:
                res.append(self.unknown)

        res.append(self.eos)

        for _ in range(self.maxLen - len(res)):
            res.append(self.pad)

        if len(data) > self.maxLen:
            res = res[:self.maxLen]

        return res

    def decode(self, x: list[int]) -> str:
        res = []

        for i in x:
            try:
                res.append(self.vocab[i])
            except:
                res.append("<UNKNOWN>")

        return " ".join(res)

    def __len__(self):
        return len(self.vocab)