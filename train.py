from timm.models.efficientnet import *
import torch
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from utils import CustomDataset, get_model, main, LabelSmoothingLoss
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torchmetrics
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1, AUROC
from pytorch_lightning.callbacks import LearningRateMonitor
import argparse
import torchvision.models as models
import ttach as tta
from pytorch_lightning.utilities.seed import seed_everything
#import warnings
#warnings.filterwarnings("ignore")



def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--fold", type=int, default=0, help="fold number: 0, 1, 2, 3, 4")
    parser.add_argument('--model_name', type=str, default='densenet', help='model name: b7, v2s, v2m, v2l')
    parser.add_argument('--file', type=str, default='best_model.ckpt', help='file_name')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument("--transform", type=str, default='./transforms/train1.yml')
    parser.add_argument("--val_transform", type=str, default='./transforms/val1.yml')
    parser.add_argument("--sampler", type=str, default='')
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument("--label_smooth", type=float, default=0.0)
    args = parser.parse_args()
    return args


class Model(pl.LightningModule):
    def __init__(self, transform='./transforms/chexclusion_val.yml', val_transform='./transforms/chexclusion_val.yml',
                 model_name='densenet', fold=0, batch_size=64, scheduler_patience=4, sampler='', label_smooth=0.0):
        super().__init__()
        self.model_name = model_name
        self.model = get_model(self.model_name)
        self.f1 = F1(num_classes=3)
        self.accuracy_train = Accuracy(num_classes=3)
        self.val_f1 = F1(num_classes=3)
        self.accuracy_val = Accuracy(num_classes=3)
        self.test_f1 = F1(num_classes=3)
        self.accuracy_test = Accuracy(num_classes=3)
        self.batch_size = batch_size
        self.transform = transform
        self.val_transform = val_transform
        self.scheduler_patience = scheduler_patience
        self.fold = fold
        #self.criterion = torch.nn.CrossEntropyLoss()
        self.criterion = LabelSmoothingLoss(classes=3, smoothing=label_smooth)
        self.soft = torch.nn.Softmax(dim=1)
        self.sampler = sampler
        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=5e-5)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=self.scheduler_patience, factor=0.1, min_lr=1e-8, verbose=True),
            'name': 'my_scheduler',
            "monitor": 'val_loss'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)#, y[:, None].float())
        self.log('train_loss', loss, on_step=True)
        self.f1(self.soft(y_hat).detach().cpu().squeeze(), y.detach().cpu())
        self.accuracy_train(self.soft(y_hat).detach().cpu().squeeze(), y.detach().cpu())
        self.log('F1', self.f1, on_step=True, prog_bar=True)
        self.log('acc', self.accuracy_train, on_step=True, prog_bar=True)
        return loss

    def train_dataloader(self):
        train_transform = A.load(self.transform, data_format='yaml')
        data_train = CustomDataset(mode='train', fold=self.fold, transform=train_transform)
        if self.sampler:
            samples_weight = []
            print('sampler will be used')
            df = pd.read_csv(f'./folds/train{self.fold}.csv')
            weight = {0: 2, 1: 1, 2: 1}
            for i in list(df.class_num):
                samples_weight.append(weight[i])
            samples_weight = torch.tensor(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
            train_loader = DataLoader(data_train, batch_size=self.batch_size, sampler=sampler, num_workers=8)
        else:
            train_loader = DataLoader(data_train, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return train_loader

    def val_dataloader(self):
        val_transform = A.load(self.val_transform, data_format='yaml')
        data_val = CustomDataset(mode='val', fold=self.fold, transform=val_transform)
        val_loader = DataLoader(data_val, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return val_loader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)#y[:, None].float())
        self.log('val_loss', val_loss, on_step=True)
        self.val_f1(self.soft(y_hat).detach().cpu().squeeze(), y.detach().cpu())
        self.accuracy_val(self.soft(y_hat).detach().cpu().squeeze(), y.detach().cpu())
        self.log('F1_val', self.val_f1, on_step=True, prog_bar=True)
        self.log('acc_val', self.accuracy_val, on_step=True, prog_bar=True)

    def test_dataloader(self):
        val_transform = A.load(self.val_transform, data_format='yaml')
        data_test = CustomDataset(mode='test', transform=val_transform)
        test_loader = DataLoader(data_test, batch_size=1, shuffle=False, num_workers=8)
        return test_loader

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        return {'pred': y_hat}

    def test_epoch_end(self, outputs):
        pred = torch.cat([x["pred"] for x in outputs])
        pred = self.soft(pred)
        df_test = pd.read_csv(f'./folds/test.csv')
        ids = []
        class_num = []
        predictions = torch.argmax(pred, dim=1)
        for idx, file in enumerate(list(df_test.file)):
            ids.append(file.split('.')[0])
            if predictions[idx] == 2:
                class_num.append(3)
            else:
                class_num.append(predictions[idx].item())
        df_test_pred = pd.DataFrame(list(zip(ids, class_num)),
                                    columns=['id', 'class'])
        df_test_pred.to_csv(f'./{self.model_name}_f{self.fold}.csv', index=False)

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop("v_num", None)
        return tqdm_dict


if __name__ == '__main__':
    seed_everything(42)
    args = parser()
    print(f'transform = {args.transform}, fold = {args.fold}, scheduler patience = {args.scheduler_patience}, early_stopping = {args.patience}')
    print(f'label smoothing = {args.label_smooth}')
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                        patience=args.patience, verbose=True, mode="min")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="acc_val",
        mode="max",
        filename="{fold}--{epoch:02d}-{AUC_val:.3f}--{F1_val:.3f}---{AUC:.3f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(fast_dev_run=False, gpus=1,
                         callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
                         min_epochs=1, max_epochs=args.epochs)
    model = Model(fold=args.fold, transform=args.transform, val_transform=args.val_transform,
                  model_name=args.model_name, batch_size=args.batch_size,
                  scheduler_patience=args.scheduler_patience, sampler=args.sampler, label_smooth=args.label_smooth)
    trainer.fit(model)
    trainer.test(ckpt_path="best")

    file_name = f"example{args.fold}.ckpt"
    trainer.save_checkpoint(args.file)
    model = Model.load_from_checkpoint(f'./{args.file}')
    if args.transform == './transforms/train1.yml':
        transform_name = 'train1'
    elif args.transform == './transforms/train680.yml':
        transform_name = 'train680'
    elif args.transform == './transforms/train1024.yml':
        transform_name = 'train1024'

    val_transform = A.load(args.val_transform, data_format='yaml')
    acc_train, f1_train, acc_val, f1_val = main(model.model, val_transform, batch_size=args.batch_size, fold=args.fold)
    if acc_train >= 0.8 and acc_val >= 0.8:
        torch.save(model.model.state_dict(), f'./best_models/{args.model_name}_train{acc_train:.3f}_val{acc_val:.3f}_f{args.fold}{args.patience}{args.scheduler_patience}{transform_name}smooth{args.label_smooth}sampler.pth')
    else:
        print(f'train_acc = {acc_train:.3f}, val_acc = {acc_val:.3f}')
        print('We will not save the model')



