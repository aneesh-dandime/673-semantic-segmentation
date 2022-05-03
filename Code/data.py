import os
import torch
import numpy as np
from typing import List, Tuple, Union
from torch.utils.data import Dataset
from cv2 import cv2
from torch.utils.data.dataloader import DataLoader

class KittiDataset(Dataset):
    def __init__(self, path: str, shape: Union[None, Tuple[int, int]] = None) -> None:
        if not os.path.exists(path):
            raise ValueError(f'The path {path} does not exist!')

        self.path = path
        self.shape = shape
        self.input_paths = os.path.join(self.path, 'image_2')
        self.gt_paths = os.path.join(self.path, 'semantic_rgb')
        self.image_names = self.__get_image_names()

    def __get_image_names(self) -> List[str]:
        image_names = []
        for image_name in os.listdir(self.input_paths):
            if not image_name.endswith('.png'):
                continue
            image_names.append(image_name)
        return image_names

    def __len__(self) -> int:
        return len(self.image_names)

    def __normalize(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(float)
        return cv2.normalize(arr, None, 0.0, 1.0, cv2.NORM_MINMAX)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_path = os.path.join(self.input_paths, self.image_names[idx])
        gt_path = os.path.join(self.gt_paths, self.image_names[idx])
        inp = cv2.imread(input_path)
        gt = cv2.imread(gt_path)
        if self.shape:
            inp = cv2.resize(inp, self.shape)
            gt = cv2.resize(gt, self.shape)
        inp = self.__normalize(inp)
        gt = self.__normalize(gt)
        return torch.from_numpy(inp).permute(2, 0, 1), torch.from_numpy(gt).permute(2, 0, 1)

if __name__ == '__main__':
    dataset = KittiDataset('Data/kitti/training', shape=(621, 187))
    dataloader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)
    iterator = iter(dataloader)
    inp, gt = iterator.next()
    print(inp.shape, gt.shape)
