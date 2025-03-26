from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.num_graphs = len(data)

    def __len__(self):
        return self.num_graphs

    def __getitem__(self, idx):
        return self.data[idx]