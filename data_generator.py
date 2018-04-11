import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import random

import mnist


class DataGenerator(Dataset):
    def __init__(self, config, phase='train'):
        self.config = config
        self.imgs, self.lbls = mnist.read(dataset=phase, path=self.config.mnist_dir)

    def __getitem__(self, idx):
        img = self.imgs[idx].astype(np.float32)
        ih, iw = img.shape
        img = (img/255. - self.config.mnist_mean) / self.config.mnist_std
        img_pad = np.zeros([3, self.config.img_size, self.config.img_size], dtype=np.float32)
        x, y = np.random.randint(0, self.config.img_size-ih, (2,), dtype=np.int16)
        for k in range(3):
            img_pad[k, y:y+ih, x:x+iw] = img
        lbl = self.lbls[idx:idx+1].astype(np.int64)
        return torch.from_numpy(img_pad), torch.from_numpy(lbl)

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    from config import Config
    from torch.utils.data import DataLoader

    config = Config()

    data_dataset = DataGenerator(config, phase='train')
    data_loader = DataLoader(data_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=1,
                              pin_memory=True)
    for i, (imgs, lbls) in enumerate(data_loader):
        img = imgs[0].numpy()
        # lbl = lbls[0].numpy()
        # print(img.shape, lbl)
        img_ori = np.uint8((img * config.mnist_std + config.mnist_mean) * 255).transpose(1, 2, 0)
        cv2.imwrite('/home/storage/lsy/interpret_cnn/tmp/eg.png', img_ori)
        break