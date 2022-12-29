import numpy
import torch
from torch import nn

def BackDoorAdjust(list_pred, list_prior):
    pred = torch.zeros_like(list_pred[0])
    # print(pred.shape) # B x class
    for i in range(len(list_pred)):
        pred += list_pred[i] * list_prior[i]
    return pred