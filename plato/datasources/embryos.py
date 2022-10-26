from plato.datasources import base
import os
import pandas as pd
from PIL import Image as Image
from typing import Callable, Tuple, Any, Optional
from torchvision.datasets import VisionDataset
import numpy as np
import torch
import numpy
from torchvision import datasets, transforms


class EmbryosDataset(VisionDataset):

    def __init__(
        self, loaded_data, targets, transform: Optional[Callable] = transforms.ToTensor(), target_transform: Optional[Callable] = None) \
            -> None:
        super().__init__(root="", transform=transform, target_transform=target_transform)
        self.loaded_data = loaded_data
        self.transform = transform
        self.target_transform = target_transform
        self.targets = targets
        self.classes = [0, 1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.loaded_data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class DataSource(base.DataSource):
    def __init__(self, client_id=0):
        super().__init__()
        self.trainset = None
        self.testset = None
        self._root = "/mnt/data/mlr_ahj_datasets/vitrolife/dataset/"

        metadata_file_path = os.path.join(self._root, "metadata.csv")
        self._meta_data = pd.read_csv(metadata_file_path)

        # Split in train and validation
        meta_data_train_validation = self._meta_data.loc[self._meta_data['Testset'] == 0]
        meta_data_test = self._meta_data.loc[self._meta_data['Testset'] == 1]
        client_train_data = meta_data_train_validation.loc[meta_data_train_validation['LabID'] == client_id - 1]
        client_test_data = meta_data_test.loc[meta_data_test['LabID'] == client_id - 1]

        train_data, train_targets, _ = self._load_data(client_train_data)
        test_data, test_targets, _ = self._load_data(client_test_data)

        self.trainset = EmbryosDataset(loaded_data=train_data, targets=train_targets)
        self.testset = EmbryosDataset(loaded_data=test_data, targets=test_targets)

    def _load_data(self, meta_data):
        data = []
        for index, row in meta_data.iterrows():
            try:
                file_path = os.path.join(self._root, "{:05d}.npz".format(row['SampleID']))
                img = self._load_image(file_path)
                data.insert(index, img)
            except:
                print(f"Cannot load id: {index}")
                meta_data.drop(index=index, inplace=True)
        label_tensor = torch.LongTensor(meta_data["Label"].tolist())
        clinic_ids = np.array(meta_data["LabID"].tolist())
        return data, label_tensor, clinic_ids


    def _load_image(self, path):
        file_data = np.load(path)
        images = file_data['images']

        focal = 1
        frame = 0
        img_raw = images[frame, :, :, focal]
        img = Image.fromarray(img_raw)
        newsize = (250, 250)
        img = img.resize(newsize)
        img_raw = np.asarray(img)
        img_raw = img_raw.astype('float32') / 255
        return img_raw

