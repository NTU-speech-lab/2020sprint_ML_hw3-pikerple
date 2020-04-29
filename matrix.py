import os
import numpy as np
#import cv2
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
# from resources.plotcm import plot_confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
import time
from torchsummary import summary

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
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 11),
        )    
    def forward(self, x):
        # nn1 = x.view(-1,3*128*128)
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

model_best = torch.load('./net_test.pkl')
# model_best.load_state_dict(torch.load('./net_test_params.pkl'))
print(model_best)
# summary(model_best, input_size=(3, 128, 128))
# model_best  = torch.load('./net_2params.pkl')
print("loading_x")
val_x = np.load('val_x.npy')

print("loading_y")
val_y = np.load('val_y.npy')
test_x = np.load('test_x.npy')
train_x = np.load('train_x.npy')
train_y = np.load('train_y.npy')

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
val_set = ImgDataset(val_x, val_y, test_transform)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
train_set = ImgDataset(train_x, train_y, test_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
# model_best.eval()
cmt = torch.zeros(10,10, dtype=torch.int64)

prediction = []
real = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        val_pred = model_best(data[0].cuda())
        val_label = np.argmax(val_pred.cpu().data.numpy(), axis=1)
        real_label = data[1].numpy()
        for y in real_label:
             real.append(y)
        for y in val_label:
             prediction.append(y)
test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
model_best.eval()
#  # model.eval()
# prediction = []
# with torch.no_grad():
#     for i, data in enumerate(test_loader):
#         test_pred = model_best(data.cuda())
#         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
#         for y in test_label:
#             prediction.append(y)
print(prediction)
print(real)

# prediction = []
# real = []
# with torch.no_grad():
#     for i, data in enumerate(train_loader):
#         val_pred = model_best(data[0].cuda())
#         val_label = np.argmax(val_pred.cpu().data.numpy(), axis=1)
#         real_label = data[1].numpy()
#         for y in real_label:
#              real.append(y)
#         for y in val_label:
#              prediction.append(y)
# test_set = ImgDataset(test_x, transform=test_transform)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
# model_best.eval()
#  # model.eval()
# prediction = []
# with torch.no_grad():
#     for i, data in enumerate(test_loader):
#         test_pred = model_best(data.cuda())
#         test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
#         for y in test_label:
#             prediction.append(y)
print(prediction)
print(real)

cm = confusion_matrix(real, prediction)
print(cm)

def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
names = (
	'Bread', 
	'Dairy product', 
	'Dessert', 
	'Egg', 
	'Fried food', 
	'Meat',
	'Noodles/Pasta', 
	'Rice', 
	'Seafood',
	'Soup', 
	'Vegetable/Fruit'
)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm, names)
plt.savefig('matrix.png')
