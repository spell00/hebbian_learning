import torch
class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]
