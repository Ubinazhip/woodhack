from timm.models.efficientnet import *
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from utils import CustomDataset, get_model, main
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, AUROC
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import torchvision.models as models
import ttach as tta
from pytorch_lightning.utilities.seed import seed_everything
import os
from tqdm import tqdm
import ttach as tta


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str, default='', help='path to the model1')
    parser.add_argument('--model2', type=str, default='', help='path to the model2')
    parser.add_argument('--model3', type=str, default='', help='path to the model3')
    parser.add_argument('--model4', type=str, default='', help='path to the model4')
    parser.add_argument('--model5', type=str, default='', help='path to the model5')
    parser.add_argument('--model6', type=str, default='', help='path to the model6')
    parser.add_argument('--model7', type=str, default='', help='path to the model7')
    parser.add_argument('--model8', type=str, default='', help='path to the model8')
    parser.add_argument('--model9', type=str, default='', help='path to the model9')
    parser.add_argument('--model10', type=str, default='', help='path to the model10')
    parser.add_argument('--model11', type=str, default='', help='path to the model11')
    parser.add_argument('--model12', type=str, default='', help='path to the model12')
    parser.add_argument("--val_transform", type=str, default='./transforms/resize680.yml')
    parser.add_argument("--file", type=str, default='ensemble', help='predicted csv file {args.file}{train_acc}{val_acc}.csv')
    parser.add_argument("--tta", type=str, default='')
    parser.add_argument("--tta2", type=str, default='')
    parser.add_argument("--crop_size", type=int, default=640)
    args = parser.parse_args()
    return args


def validate(models, criterion, val_loader):
    for model in models:
        model.eval()
    tqdm_loader = tqdm(val_loader)
    f1 = F1(num_classes=3)
    accuracy = Accuracy(num_classes=3)
    soft = torch.nn.Softmax(dim=1)
    average_loss = 0

    for idx, (X, y) in enumerate(tqdm_loader):
        X, y = X.cuda(), y.cuda()
        with torch.no_grad():
            y_hat = 0
            for model in models:
                y_hat += model(X) / (len(models))
            loss = criterion(y_hat, y)
            average_loss = (average_loss * idx + loss) / (idx + 1)
            f1(soft(y_hat).detach().cpu().squeeze(), y.detach().cpu())
            accuracy(soft(y_hat).detach().cpu().squeeze(), y.detach().cpu())
    f1_val = f1.compute()
    acc = accuracy.compute()
    f1.reset()
    accuracy.reset()
    return acc


def ensemble(models, val_loaders, test_loader, criterion,file_name='ensemble'):
    acc_vals = []
    for i, val_loader in enumerate(val_loaders):
        print(f'validation: fold = {i}')
        acc = validate(models, criterion, val_loader)
        acc_vals.append(acc)
    for idx, acc in enumerate(acc_vals):
        print(f'fold = {idx}: accuracy = {acc:.3f}')
    print(f'mean is {(sum(acc_vals) / len(acc_vals)):.3f}')
    for model in models:
        model.eval()
    tqdm_loader = tqdm(test_loader)
    soft = torch.nn.Softmax(dim=1)
    df_test = pd.read_csv(f'./folds/test.csv')
    for idx, (X) in enumerate(tqdm_loader):
        with torch.no_grad():
            y_hat = 0
            for model in models:
                y_hat += soft(model(X))/(len(models))
            if idx == 0:
                predictions = y_hat
            else:
                predictions = torch.cat([predictions, y_hat])
    ids = []
    class_num = []
    predictions = torch.argmax(predictions, dim=1)
    for idx, file in enumerate(list(df_test.file)):
        ids.append(file.split('.')[0])
        if predictions[idx] == 2:
            class_num.append(3)
        else:
            class_num.append(predictions[idx].item())
    df_test_pred = pd.DataFrame(list(zip(ids, class_num)),
                                columns=['id', 'class'])
    df_test_pred.to_csv(f'./{file_name}_accf0{acc_vals[0]}f1{acc_vals[1]}f2{acc_vals[2]}f3{acc_vals[3]}f4{acc_vals[4]}.csv', index=False)


if __name__ == '__main__':
    args = parser()
    models = []
    crop_size = args.crop_size
    transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.VerticalFlip(),
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ])
    if args.tta:
        print('test time augmentation will be used')
    else:
        print('tta will not be used')
    if args.model1:
        model = get_model(args.model1.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model1))
        if args.tta == 'ten_crop_transform':
            print(f'crop size is {crop_size}')
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta == 'transforms':
            print('horizontal, vertical flips will be used during tta')
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model2:
        model = get_model(args.model2.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model2))
        if args.tta == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model3:
        model = get_model(args.model3.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model3))
        if args.tta == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model4:
        model = get_model(args.model4.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model4))
        if args.tta == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model5:
        model = get_model(args.model5.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model5))
        if args.tta == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model6:
        model = get_model(args.model6.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model6))
        if args.tta == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model7:
        model = get_model(args.model7.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model7))
        if args.tta2 == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta2 == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model8:
        model = get_model(args.model8.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model8))
        if args.tta2 == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta2 == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model9:
        model = get_model(args.model9.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model9))
        if args.tta2 == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta2 == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model10:
        model = get_model(args.model10.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model10))
        if args.tta2 == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta2 == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model11:
        model = get_model(args.model11.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model11))
        if args.tta2 == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta2 == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)
    if args.model12:
        model = get_model(args.model12.split('/')[-1].split('_')[0])
        model.load_state_dict(torch.load(args.model12))
        if args.tta2 == 'ten_crop_transform':
            model = tta.ClassificationTTAWrapper(model,# transforms=transforms,
                                                  tta.aliases.ten_crop_transform(crop_height=crop_size, crop_width=crop_size),
                                                 merge_mode='mean')
        elif args.tta2 == 'transforms':
            model = tta.ClassificationTTAWrapper(model,  transforms=transforms)
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        models.append(model)

    val_transform = A.load(args.val_transform, data_format='yaml')
    print(f'val transform = {args.val_transform} is loaded, ensemble of {len(models)}  models')
    data_test = CustomDataset(mode='test', transform=val_transform)
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=8)

    val_loaders = []

    for i in range(0, 5):
        data_val = CustomDataset(mode='val',fold=i, transform=val_transform)
        val_loader = DataLoader(data_val, batch_size=4, shuffle=False, num_workers=8)
        val_loaders.append(val_loader)

    ensemble(models, val_loaders=val_loaders, test_loader=test_loader, criterion=torch.nn.CrossEntropyLoss(), file_name=args.file)




