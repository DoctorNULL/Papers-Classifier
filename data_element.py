class DataElement(object):
    def __init__(self, abstract: list[str], labels:list[int]):
        self.abstract = abstract
        self.labels = labels

    def __str__(self):
        return str(self.abstract) + "\t" + str(self.labels)

    def __repr__(self):
        return str(self)