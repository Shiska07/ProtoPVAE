import os
import pytorch_lightning as pl
from torchvision import transforms
from utils.preprocess import mean, std
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from configs.proto_configs import input_height
from configs.train_settings import use_validation

class CUBDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir,
                 batch_size,
                 push_batch_size=80,
                 num_workers=4):

        super().__init__()

        self.data_dir = data_dir

        self.batch_size = batch_size
        self.push_batch_size = push_batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([transforms.Resize(size=(
            input_height, input_height)), transforms.ToTensor(),
                                 transforms.Normalize(mean=mean,
                                 std=std)])
        if use_validation:
            self.train_dir = os.path.join(self.data_dir,
                                      "cub200_cropped", "train_cropped_augmented_80")
            self.val_dir = os.path.join(self.data_dir, "cub200_cropped",
                                        "val_cropped_augmented_20")
            # load the dataset
            self.train_dataset = datasets.ImageFolder(self.train_dir,
                                                      transform=self.transform)
            self.val_dataset = datasets.ImageFolder(self.val_dir,
                                                    transform=self.transform)
        else:
            self.train_dir = os.path.join(self.data_dir,
                                      "cub200_cropped", "train_cropped_augmented")

            # load the dataset
            self.train_dataset = datasets.ImageFolder(self.train_dir,
                                                      transform=self.transform)

        self.test_dir = os.path.join(self.data_dir, "cub200_cropped",
                                     "test_cropped")
        self.test_dataset = datasets.ImageFolder(self.test_dir,
                                                 transform=self.transform)

        self.train_push_dir = os.path.join(self.data_dir,
                                      "cub200_cropped", "train_cropped")

        self.train_push_dataset = datasets.ImageFolder(self.train_push_dir,
                                                        transforms.Compose([
                                                        transforms.Resize(size=(
                                                        input_height, input_height)),
                                                        transforms.ToTensor()
                                                        ]))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          pin_memory=True, persistent_workers=True, shuffle=True,
                          num_workers=self.num_workers)

    def train_push_dataloader(self):
        return DataLoader(self.train_push_dataset, batch_size=self.push_batch_size,
                          pin_memory=True, persistent_workers=True,
                           num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          pin_memory=True, persistent_workers=True,
                           drop_last=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          pin_memory=True, persistent_workers=True,
                          num_workers=self.num_workers)

