from torch.utils.data import Dataset
import numpy as np
import os
import glob
import matplotlib.image as mpimg
import cv2
import torch
import pandas as pd
import matplotlib.image as mpimg
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from timm.models.efficientnet import *
import torchvision.models as models
from torch.utils.data import DataLoader
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, AUROC
import timm
from timm.models.gluon_resnet import *
import torch.nn as nn
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, fold=0, mode='train', transform=None):
        self.mode = mode
        if mode == 'train':
            self.df = pd.read_csv(f'./folds/train{fold}.csv')
            self.root = '../dataset_drove/train/train'
        elif mode == 'val':
            self.df = pd.read_csv(f'./folds/val{fold}.csv')
            self.root = '../dataset_drove/train/train'
        else:
            self.df = pd.read_csv(f'./folds/test.csv')
            self.root = self.root = '../dataset_drove/test/test'
        if transform:
            self.transform = transform
        else:
            self.transform = A.Compose([
                            A.Resize(512, 512),
                            ToTensorV2()
                            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        file_name = row['file']
        if self.mode == 'test':
            file_path = os.path.join(self.root, file_name)
        else:
            class_name = row['class_name']
            file_path = os.path.join(self.root, class_name, file_name)

        img = mpimg.imread(file_path)#cv2

        if self.transform:
            img = self.transform(image=img)['image']

        if self.mode == 'test':
            return img
        else:
            return img, row['class_num']


def get_model(model_name):
    if model_name == 'densenet':
        #model = models.densenet121(pretrained=True)
        #model.classifier = nn.Linear(in_features=1024, out_features=3, bias=True)
        model = timm.create_model('densenet121', pretrained=True, num_classes=3)
    elif model_name == 'densenet169':
        model = timm.create_model('densenet169', pretrained=True, num_classes=3)
    elif model_name == 'xception':
        model = timm.create_model('xception', pretrained=True, num_classes=3)
    elif model_name == 'b1':
        model = tf_efficientnet_b1(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        model.classifier = nn.Linear(in_features=1280, out_features=3, bias=True)
    elif model_name == 'b3':
        model = tf_efficientnet_b3(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)#10697769
        model.classifier = nn.Linear(in_features=1536, out_features=3, bias=True)
    elif model_name == 'v2s':
        model = tf_efficientnetv2_s_in21ft1k(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        model.classifier = nn.Linear(in_features=1280, out_features=3, bias=True)
    elif model_name == 'b2':
        model = tf_efficientnet_b2(pretrained=True, drop_rate=0.3, drop_path_rate=0.2)
        model.classifier = nn.Linear(in_features=1408, out_features=3, bias=True)
    elif model_name =='mobilev3':
        model = timm.create_model('mobilenetv3_large_100', num_classes=3, pretrained=True)
    elif model_name == 'vit_base_patch32_384':
        model = timm.create_model('vit_base_patch32_384', pretrained=True, num_classes=3)
    elif model_name == 'vit_tiny_patch16_384':
        model = timm.create_model('vit_tiny_patch16_384', pretrained=True, num_classes=3)
    elif model_name == 'gluon':
        model = gluon_seresnext50_32x4d(pretrained=True)
        model.fc = nn.Linear(2048, 3)
    else:
        print('No such model in our zooo')
        model_name = 'NO SUCH MODEL'

    print(f'The model {model_name} is ready!')
    return model


def validate(model, loader, criterion):
    model.eval()
    tqdm_loader = tqdm(loader)
    f1 = F1(num_classes=3)
    accuracy = Accuracy(num_classes=3)
    soft = torch.nn.Softmax(dim=1)
    average_loss = 0

    for idx, (X, y) in enumerate(tqdm_loader):
        with torch.no_grad():
            y_hat = model(X)
            loss = criterion(y_hat, y)
            average_loss = (average_loss * idx + loss) / (idx + 1)
            f1(soft(y_hat).detach().cpu().squeeze(), y.detach().cpu())
            accuracy(soft(y_hat).detach().cpu().squeeze(), y.detach().cpu())
    f1_val = f1.compute()
    acc = accuracy.compute()
    f1.reset()
    accuracy.reset()

    return acc, f1_val


def main(model, val_transform, batch_size=8, fold=0):
    data_train = CustomDataset(mode='train', fold=fold, transform=val_transform)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
    data_val = CustomDataset(mode='val', fold=fold, transform=val_transform)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=8)

    criterion = torch.nn.CrossEntropyLoss()

    acc_train, f1_train = validate(model, train_loader, criterion)

    print(f'train: accuracy {acc_train:.3f} and f1 = {f1_train:.3f}')

    acc_val, f1_val = validate(model, val_loader, criterion)

    print(f'val: accuracy {acc_val:.3f} and f1 = {f1_val:.3f}')

    return acc_train, f1_train, acc_val, f1_val


class LabelSmoothingLoss(nn.Module):
    '''
    https://github.com/pytorch/pytorch/issues/7455
    '''
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))





