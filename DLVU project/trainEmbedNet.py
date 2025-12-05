#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import time
import os
import argparse
import pdb
import glob
import datetime
import numpy
import logging
from EmbedNet import *
from DatasetLoader import get_data_loader
from sklearn import metrics
import torchvision.transforms as transforms

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Parse arguments
# ## ===== ===== ===== ===== ===== ===== ===== =====

parser = argparse.ArgumentParser(description = "Face Recognition Training");

## Data loader
parser.add_argument('--batch_size',         type=int, default=100,	help='Batch size, defined as the number of classes per batch');
parser.add_argument('--max_img_per_cls',    type=int, default=500,	help='Maximum number of images per class per epoch');
parser.add_argument('--nDataLoaderThread',  type=int, default=5, 	help='Number of data loader threads');

## Training details (추가함)
parser.add_argument('--test_interval',  type=int,   default=2,      help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=10,     help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="softmax",  help='Loss function to use');
parser.add_argument('--aug',            type=str,   default="basic",
                    help='Data augmentation type: basic / strong')


## Optimizer (momentum 추가함)
parser.add_argument('--optimizer',      type=str,   default="adam", help='Optimizer');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Initial learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.90,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=0,      help='Weight decay in the optimizer');
parser.add_argument('--momentum',       type=float, default=0.9,    help='Momentum for SGD optimizer')


## Loss functions
parser.add_argument('--margin',         type=float, default=0.1,    help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=30,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerClass',      type=int,   default=1,      help='Number of images per class per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=9500,   help='Number of classes in the softmax layer, only for softmax-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights, otherwise initialise with random weights');
parser.add_argument('--save_path',      type=str,   default="exps/exp1", help='Path for model and logs');

## Training and evaluation data
parser.add_argument('--train_path',     type=str,   default="data/train",   help='Absolute path to the train set');
parser.add_argument('--train_ext',      type=str,   default="jpg",  help='Training files extension');
parser.add_argument('--test_path',      type=str,   default="data/val",     help='Absolute path to the test set');
parser.add_argument('--test_list',      type=str,   default="data/val_pairs.csv",   help='Evaluation list');

## Model definition
parser.add_argument('--model',          type=str,   default="ResNet18", help='Name of model definition');
parser.add_argument('--nOut',           type=int,   default=512,    help='Embedding size in the last FC layer');

## For test only
parser.add_argument('--eval',           dest='eval', action='store_true',   help='Eval only')
parser.add_argument('--output',         type=str,   default="",     help='Save a log of output to this file name');

## Training
parser.add_argument('--gpu',            type=int,   default=9,      help='GPU index');

args = parser.parse_args();

## ===== ===== ===== ===== ===== ===== ===== =====
## Script to compute EER
## ===== ===== ===== ===== ===== ===== ===== =====

def compute_eer(all_labels,all_scores):

    # compute receiver operating characteristic (ROC) for binary classification
    # pos_label=1 : 같은 사람일 때 1, 다른 사람일 때 0이라고 가정
    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_scores, pos_label=1)

    # calculate false negative rate (FNR)
    fnr = 1 - tpr

    # calculate equal error rate (EER). The EER is the error rate at which FNR is equal to FPR.
    # FNR와 FPR의 차이가 가장 작은 지점을 찾고, 그 때의 FPR을 EER로 사용
    idxE = numpy.nanargmin(numpy.absolute(fnr - fpr))
    EER = fpr[idxE]

    return EER


## ===== ===== ===== ===== ===== ===== ===== =====
## Trainer script
## ===== ===== ===== ===== ===== ===== ===== =====

def main_worker(args):

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.save_path+"/scores.txt", mode="a+"),
        ],
        level=logging.DEBUG,
        format='[%(levelname)s] :: %(asctime)s :: %(message)s', 
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ## Load models(중요!)
    model = EmbedNet(**vars(args)).cuda();

    ep          = 1

    ## Input transformations for training (you can change if you like)
    
    if args.aug == "basic":
        train_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize(256),
            transforms.RandomCrop([224,224]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    elif args.aug == "strong":
        # Config D 용 : ColorJitter + 약한 RandomErasing
        train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop([224,224]),
                transforms.ColorJitter(brightness=0.2,
                                       contrast=0.2,
                                       saturation=0.2,
                                       hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.4, scale=(0.02, 0.2),
                                         ratio=(0.3, 3.3), value=0),
            ])
    else:
        raise ValueError(f"Unknown aug type: {args.aug}")



    ## Input transformations for evaluation (you can change if you like)
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(256),
         transforms.CenterCrop([224,224]),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    ## Initialise trainer and data loader
    trainLoader = get_data_loader(transform=train_transform, **vars(args));
    trainer     = ModelTrainer(model, **vars(args))

    ## Load model weights
    modelfiles = glob.glob(f'{args.save_path}/epoch0*.model')
    modelfiles.sort()

    ## If the target directory already exists, start from the existing file
    if len(modelfiles) >= 1:
        trainer.loadParameters(modelfiles[-1]);
        print(f"Model {modelfiles[-1]} loaded from previous state!");
        ep = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
    elif(args.initial_model != ""):
        trainer.loadParameters(args.initial_model);
        print(f"Model {args.initial_model} loaded!");

    ## If the current iteration is not 1, update the scheduler
    for ii in range(1,ep):
        trainer.__scheduler__.step()

    ## Print total number of model parameters
    pytorch_total_params = sum(p.numel() for p in model.__E__.parameters())
    print(f'Total model parameters: {pytorch_total_params:,}')
    
    ## Evaluation code 
    if args.eval == True:

        sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
        
        EER = compute_eer(lab, sc)

        print(f'EER {EER*100:.2f}%')

        if args.output != '':
            with open(args.output,'w') as f:
                for ii in range(len(sc)):
                    f.write(f'{sc[ii]:.4f},{lab[ii]:d},{trials[ii]}\n')

        quit();

    ## Log arguments
    logger.info(f'{args}')

    ## Core training script
    for ep in range(ep,args.max_epoch+1):

        clr = [x['lr'] for x in trainer.__optimizer__.param_groups]

        logger.info(f"Epoch {ep:04d} started with LR {max(clr):.5f} ");
        loss = trainer.train_network(trainLoader);
        logger.info(f"Epoch {ep:04d} completed with TLOSS {loss:.5f}");

        if ep % args.test_interval == 0:
            
            sc, lab, trials = trainer.evaluateFromList(transform=test_transform, **vars(args))
            EER = compute_eer(lab, sc)

            logger.info(f"Epoch {ep:04d}, Val EER {EER*100:.2f}%");
            trainer.saveParameters(args.save_path+f"/epoch{ep:04d}.model");

# ## ===== ===== ===== ===== ===== ===== ===== =====
# ## Main function
# ## ===== ===== ===== ===== ===== ===== ===== =====


def main():

    os.environ["CUDA_VISIBLE_DEVICES"]=f'{args.gpu}'
            
    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)

    main_worker(args)


if __name__ == '__main__':
    main()