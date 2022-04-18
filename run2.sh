#!/bin/sh

CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name densenet --batch_size 8 --epochs 30 --fold 0 --patience 7 --scheduler_patience 4 --sampler sampler --transform ./transforms/train680.yml --val_transform ./transforms/val680.yml --label_smooth 0.1
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name densenet --batch_size 8 --epochs 30 --fold 1 --patience 7 --scheduler_patience 4 --sampler sampler --transform ./transforms/train680.yml --val_transform ./transforms/val680.yml --label_smooth 0.1
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name densenet --batch_size 8 --epochs 30 --fold 2 --patience 7 --scheduler_patience 4 --sampler sampler --transform ./transforms/train680.yml --val_transform ./transforms/val680.yml --label_smooth 0.1
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name densenet --batch_size 8 --epochs 30 --fold 3 --patience 7 --scheduler_patience 4 --sampler sampler --transform ./transforms/train680.yml --val_transform ./transforms/val680.yml --label_smooth 0.1
CUDA_VISIBLE_DEVICES=1 python3 train.py --model_name densenet --batch_size 8 --epochs 30 --fold 4 --patience 7 --scheduler_patience 4 --sampler sampler --transform ./transforms/train680.yml --val_transform ./transforms/val680.yml --label_smooth 0.1


