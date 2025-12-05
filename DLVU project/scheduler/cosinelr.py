#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Scheduler(optimizer, test_interval, max_epoch, lr_decay, **kwargs):
    """
    CosineAnnealingLR version of StepLR scheduler.
    Arguments are matched exactly to steplr.py for compatibility.
    
    test_interval : ignored (kept for interface compatibility)
    lr_decay      : ignored (cosine does not use gamma)
    """

    # T_max = total number of epochs
    T_max = max_epoch

    sche_fn = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max,
        eta_min=1e-6,    # optional minimal LR
    )

    lr_step = 'epoch'

    print('Initialised Cosine LR scheduler')

    return sche_fn, lr_step
