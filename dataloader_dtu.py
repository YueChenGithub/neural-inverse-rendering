import torch
import numpy as np
import os
from pathlib import Path
from PIL import Image
import json
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from tools.color_mapping_blender import linear2srgb, srgb2linear
from tools.convert_img_channel import rgba2rgb

class Dataset_dtu(torch.utils.data.Dataset):
    def __init__(self, split, data_dir, data_name):
        assert split in ['train_', 'val_', 'test_']
        self.data_dir = data_dir
        self.data_name = data_name
        self.data_id_list = sorted([d for d in os.listdir(data_dir) if split in d])
        assert len(self.data_id_list) > 0
        self.convert_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data_id_list)

    def __getitem__(self, index):
        data_id = self.data_id_list[index]
        data_path = Path(self.data_dir, data_id)

        # read image
        image = Image.open(Path(data_path, f'{self.data_name}.png'))
        image = self.convert_tensor(image)

        # read albedo
        albedo = torch.ones_like(image)
        albedo_black = albedo


        # read metadata
        with open(Path(data_path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        return {
            'data_id': data_id,
            'image_gt': image,
            'albedo_gt': albedo,
            'albedo_gt_black': albedo_black,
            'metadata': metadata
        }


class Dataloader_dtu(LightningDataModule):
    def __init__(self, data_dir, data_name='rgb', batch_size=1, num_workers=16):
        super().__init__()
        self.data_dir = data_dir
        self.data_name = data_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.enable_val = True
        self.test_val_dataset = False

        self.test_dataset = None
        self.val_dataset = None
        self.train_dataset = None

        self.save_hyperparameters()

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = Dataset_dtu(split='train_', data_dir=self.data_dir, data_name=self.data_name)
            if self.enable_val:
                self.val_dataset = Dataset_dtu(split='train_', data_dir=self.data_dir, data_name=self.data_name)

        if stage == 'test':
            if self.test_val_dataset:
                # calculate correction on val dataset
                self.test_dataset = Dataset_dtu(split='train_', data_dir=self.data_dir, data_name=self.data_name)
            else:
                # run test on test dataset
                self.test_dataset = Dataset_dtu(split='train_', data_dir=self.data_dir, data_name=self.data_name)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                           num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                                           num_workers=self.num_workers)


