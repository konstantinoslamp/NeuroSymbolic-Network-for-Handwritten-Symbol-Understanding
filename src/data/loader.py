class DataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.dataset):
            raise StopIteration
        
        batch = self.dataset[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return batch

    def reset(self):
        self.index = 0