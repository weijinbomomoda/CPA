import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import cv2



def image_list(imageRoot, txt='list.txt'):
    f = open(txt, 'wt')
    for (label, filename) in enumerate(sorted(os.listdir(imageRoot), reverse=False)):
        if os.path.isdir(os.path.join(imageRoot, filename)):
            for imagename in os.listdir(os.path.join(imageRoot, filename)):
                name, ext = os.path.splitext(imagename)
                ext = ext[1:]
                if ext == 'jpg' or ext == 'png' or ext == 'bmp':
                    f.write('%s %d\n' % (os.path.join(imageRoot, filename, imagename), label))
    f.close()


def shuffle_split(listFile, trainFile, valFile):
    with open(listFile, 'r') as f:
        records = f.readlines()
    random.shuffle(records)
    num = len(records)
    trainNum = int(num * 0.8)
    with open(trainFile, 'w') as f:
        f.writelines(records[0:trainNum])
    with open(valFile, 'w') as f1:
        f1.writelines(records[trainNum:])


class My_Dataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            #由于新增广医附一数据和之前数据格式不一样，所以加了个IF语句做判断
            imgs.append((words[0]+' '+words[1], int(words[2])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    image_list('/mnt/wjb/data/CFPA_SAIA_SA_AN', '/mnt/wjb/data/CFPA_SAIA_SA_AN/total.txt')
    train_data = My_Dataset(txt='/mnt/wjb/data/CFPA_SAIA_SA_AN/total.txt',transform=transforms.ToTensor()) #transform=transforms.ToTensor()
    print(train_data.__len__())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
    for i, (batch_img, batch_label) in enumerate(train_loader):
        print(i)
        print(batch_img.shape)
        print(batch_label.shape)