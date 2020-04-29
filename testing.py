import matplotlib
matplotlib.use("Agg")
import sys
import os
import numpy as np
import cv2
#import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd

# from resources.plotcm import plot_confusion_matrix
from torch.utils.data import DataLoader, Dataset
import time

workspace_dir = sys.argv[1]
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(128, 128))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        #input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]
            # nn.Dropout(0.3),

            nn.Conv2d(64, 64, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
            nn.Dropout(0.25),

           
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
            nn.Dropout(0.25),

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            nn.Dropout(0.25),
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
            nn.Dropout(0.25),

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 11),
        )    
    def forward(self, x):
        # nn1 = x.view(-1,3*128*128)
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

model_best = Classifier().cuda()
  # model = torch.load('./net_2.pkl')
model_best.load_state_dict(torch.load("./net_2params.pkl"))
# model_best.load_state_dict(torch.load('./net_test_params.pkl'))
# print(model_best)
# summary(model_best, input_size=(3, 128, 128))
# model_best  = torch.load('./net_2params.pkl')


print("loading_testing_data")

test_x = readfile(os.path.join(workspace_dir, "testing"), False)

train_transform = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    #transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15), #隨機旋轉圖片
    #transforms.Grayscale(1),
    #transforms.ColorJitter(brightness=(0, 36), contrast=(0, 10), saturation=(0, 25), hue=(-0.5, 0.5)),
   
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
     # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
    
])
#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    # transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
batch_size = 128


test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model_best.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model_best(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)
#將結果寫入 csv 檔
with open(sys.argv[2], 'w') as f:
    f.write('Id,Category\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))