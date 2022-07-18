from torch.utils.data import Dataset
from torchvision.io import read_image
from utils import crop_image_to_correct_size


class ImageDataset(Dataset):

    def __init__(
        self,
        files,
        device="cpu"
    ):

        super().__init__()

        self.X = []
        for filepath in files:
            self.X.append(
                crop_image_to_correct_size(
                    read_image(filepath) / 255
                )
            )
            self.X[-1] = self.X[-1].to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
