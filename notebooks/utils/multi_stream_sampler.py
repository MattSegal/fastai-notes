# Custom Sampler
from torch.utils.data.sampler import Sampler

class MultiStreamSampler(Sampler):
    def __init__(self, data_set, batch_size):
        self.stream_size = len(data_set) // batch_size
        self.batch_size = batch_size
                
    def __len__(self):
        return self.stream_size * self.batch_size

    def __iter__(self):
        stream_offsets = [self.stream_size * i for i in range(self.batch_size)]
        count = 0
        while count < self.stream_size:
            for offset in stream_offsets:
                yield offset + count

            count += 1