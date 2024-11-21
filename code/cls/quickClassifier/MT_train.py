import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import os
import struct
import numpy as np
from models.pretrainedModel.pretrainmodels import get_resnet50,get_resnet50_CBAM,get_resnet152,get_resnet152_CBAM,get_vit, get_mymodel

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


class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
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

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--use_cuda', default=True, help='using CUDA for training')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
if args.cuda:
    torch.backends.cudnn.benchmark = True

def train(args, device):
    # 定义数据
    image_list('/mnt/wjb/data/CPA_V1', '/mnt/wjb/data/CPA_V1/total.txt')
    shuffle_split('/mnt/wjb/data/CPA_V1/total.txt', '/mnt/wjb/data/CPA_V1/train.txt', '/mnt/wjb/data/CPA_V1/val.txt')

    train_datasets = MyDataset(txt='/mnt/wjb/data/CPA_V1/train.txt', transform=transforms.ToTensor())
    test_datasets = MyDataset(txt='/mnt/wjb/data/CPA_V1/val.txt', transform=transforms.ToTensor())
    train_loader = DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_datasets, batch_size=args.batch_size)

    # 定义模型
    student_model = get_mymodel().to(device)
    mean_teacher = get_mymodel().to(device)

    # 回归器
    optimizer = optim.SGD(student_model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(args.epochs):

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.float().to(device), target.long().to(device)
            idx = torch.where(target < 20)  # 过滤target

            if idx[0].shape == torch.Size([0]): continue
            optimizer.zero_grad()
            # print(target.shape)
            # print(target_HT.shape)

            output = student_model(data)
            # 防止梯度传递到mean_teacher模型
            with torch.no_grad():
                mean_t_output = mean_teacher(data)

            # 以mean_teacher的推理结果为target, 计算student_model的均方损失误差
            const_loss = F.mse_loss(output, mean_t_output)

            # 计算总体误差
            weight = 0.2
            # 有target的样本与target进行损失计算
            loss = F.nll_loss(output[idx], target[idx]) + weight * const_loss
            # loss = F.nll_loss(output, target[idx])
            loss.backward()
            optimizer.step()

            # update mean_teacher的模型参数
            alpha = 0.95
            for mean_param, param in zip(mean_teacher.parameters(), student_model.parameters()):
                mean_param.data.mul_(alpha).add_(1 - alpha, param.data)

            # print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss.item()))
        test(student_model, device, test_loader, "student")
        test(mean_teacher, device, test_loader, "teacher")
        if (args.save_model and False):
            torch.save(student_model.state_dict(), "cpa_cnn.pt")


def test(model, device, test_loader, name):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.float().to(device), target.long().to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 计算loss
            pred = output.argmax(dim=1, keepdim=True)  # 推理结果
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('{} Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(name,
                                                                                test_loss, correct,
                                                                                len(test_loader.dataset),
                                                                                100. * correct / len(
                                                                                    test_loader.dataset)))
    model.train()


if __name__ == '__main__':
    # 配置
    parser = argparse.ArgumentParser(description='SSLpyTorch')
    parser.add_argument('--train_batch_size', type=int, default=30)
    parser.add_argument('--test_batch_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--save-model', action='store_true', default=False)
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # device = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cuda")
    # 训练
    train(args, device)
