#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFunction(nn.Module):
    """
    Additive Angular Margin Softmax (AAM-Softmax / ArcFace-style)

    - AM-Softmax:   cos(theta_y) - m
    - AAM-Softmax:  cos(theta_y + m)

    Args:
        nOut    : feature dimension (embedding size)
        nClasses: number of classes
        margin  : angular margin m (in radians, e.g. 0.2 ~ 0.5)
        scale   : scale factor s
    """
    def __init__(self, nOut, nClasses, margin=0.2, scale=30, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True
        self.m = margin
        self.s = scale

        # class weight (W): [nOut, nClasses]
        self.weight = nn.Parameter(torch.randn(nOut, nClasses))
        nn.init.xavier_normal_(self.weight, gain=1)

        self.criterion = nn.CrossEntropyLoss()

        print(f'Initialised AAM-Softmax Loss (m={self.m}, s={self.s})')

    def forward(self, x, label=None):
        """
        x: [N, nOut]  (EmbedNet에서 이미 [batch, feat_dim] 형태로 들어옴)
        label: [N]
        """

        # L2-normalize features and weights
        x_norm = F.normalize(x, dim=1)           # [N, nOut]
        W_norm = F.normalize(self.weight, dim=0) # [nOut, nClasses]

        # cosine logits: [N, C], 각 원소는 cos(theta)
        logits = torch.matmul(x_norm, W_norm)

        # 평가 시 (label 없음)에는 그대로 cos(theta)만 반환
        if label is None:
            return logits

        # 보장: label shape [N]
        label = label.view(-1).long()
        N = x.size(0)
        idx = torch.arange(0, N, device=x.device)

        # 수치 안정성을 위해 clamp
        cos_theta = logits.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # theta = arccos(cos_theta)
        theta = torch.acos(cos_theta)

        # target 클래스에만 angular margin 추가: cos(theta + m)
        target_logit = torch.cos(theta + self.m)

        # 기존 logits를 복사한 뒤, 타겟 위치만 교체
        logits_m = logits.clone()
        logits_m[idx, label] = target_logit[idx, label]

        # scale 곱해서 CE
        logits_m = logits_m * self.s
        loss = self.criterion(logits_m, label)

        return loss
