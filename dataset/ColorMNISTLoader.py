import os

import numpy as np

import torch

from torchvision import transforms
from torchvision import datasets

class ColoredMNIST(datasets.VisionDataset):
    def __init__(self, root='./data', env='train1', transform=None, target_transform=None, merge_col = False):
        super(ColoredMNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)

        if env in ['train1', 'train2', 'test']:
            data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')
        elif env == 'all_train':
            data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', 'train1.pt')) + \
                                     torch.load(os.path.join(self.root, 'ColoredMNIST', 'train2.pt'))
        else:
            raise RuntimeError(f'{env} env unknown. Valid envs are train1, train2, test, and all_train')
        # print(type(data_label_tuples))
        self.num = [0,0]
        self.col_label = np.zeros((2,2))
        self.data_label_color_tuples = []
        for img, target in data_label_tuples:
            img = transforms.ToTensor()(img)
            # print(img.shape) 
            col = (torch.sum(img[0])==0)
            self.num[col] += 1
            self.col_label[col][target] += 1
            if merge_col:
                img = img.sum(dim=0).unsqueeze(dim=0)
            self.data_label_color_tuples.append(tuple([img, target, col]))
        # self.data_label_color_tuples = torch.tensor(self.data_label_color_tuples)
    def __getitem__(self, index):
        """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
        img, target, col = self.data_label_color_tuples[index]
        # img = transforms.ToTensor()(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, col

    def __len__(self):
        return len(self.data_label_color_tuples)