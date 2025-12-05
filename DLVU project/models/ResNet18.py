#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision
import torch.nn as nn

#def MainModel(nOut=256, **kwargs):
    
#    return torchvision.models.resnet18(num_classes=nOut)

def MainModel(nOut=256, **kwargs):
    m = torchvision.models.resnet18(weights=None)
    in_feat = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_feat, nOut)
    )
    return m