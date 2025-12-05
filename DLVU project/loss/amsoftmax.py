#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.1, scale=30, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.m = margin
        self.s = scale

        # class weight (W): [nOut, nClasses]
        self.weight = nn.Parameter(torch.randn(nOut, nClasses))
        nn.init.xavier_normal_(self.weight, gain=1)

        self.criterion = nn.CrossEntropyLoss()

        print(f'Initialised AM-Softmax Loss (m={self.m}, s={self.s})')

    def forward(self, x, label=None):
        """
        x: [N, nOut]  (EmbedNet에서 이미 [batch, feat_dim] 형태로 들어옴)
        label: [N]
        """

        # L2-normalize features and weights
        x_norm = F.normalize(x, dim=1)           # [N, nOut]
        W_norm = F.normalize(self.weight, dim=0) # [nOut, nClasses]

        # cosine logits: [N, C]
        logits = torch.matmul(x_norm, W_norm)

        if label is None:
            return logits

        # 보장: label shape [N]
        label = label.view(-1).long()
        N = x.size(0)
        idx = torch.arange(0, N, device=x.device)

        # 타겟 클래스에 margin 주입 (cos(theta_yi) - m)
        logits_m = logits.clone()
        logits_m[idx, label] = logits[idx, label] - self.m

        # scale 곱해서 CE
        logits_m = logits_m * self.s
        loss = self.criterion(logits_m, label)

        return loss
