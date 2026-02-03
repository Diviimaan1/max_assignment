import torch
from torch.utils.data import Dataset

class DermatologyDataset(Dataset):
    def __init__(self, images, labels=None):
        """
        images: numpy array (N, 28, 28)
        labels: numpy array (N,) or None for test data
        """
        self.images = images.astype("float32")
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx]              # (28, 28, 3)
        x = torch.from_numpy(x).permute(2, 0, 1)  # (3, 28, 28)

        if self.labels is not None:
            y = torch.tensor(self.labels[idx], dtype=torch.long)
            return x, y

        return x
