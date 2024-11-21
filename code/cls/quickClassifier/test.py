import torch.utils.data.distributed
import torchvision.transforms as transforms
import shutil
from torch.autograd import Variable
import os
from PIL import Image

classes = ('F', 'T')

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)
path = '/mnt/wjb/dataset/Project_CPA/work_dir/train/CPA/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file)
    img=img.convert("RGB")
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
    if classes[pred.data.item()]=='T':
        shutil.copy(path+file,"/mnt/llz/dataset/Project_CPA/betweenCPA/test/T/")
    else:
        shutil.copy(path + file, "/mnt/llz/dataset/Project_CPA/betweenCPA/test/F/")


