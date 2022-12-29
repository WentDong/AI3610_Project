import torch
from tqdm import *

def eval(Model, TestLoader, rate):
    loop = tqdm(enumerate(TestLoader), total=len(TestLoader))
    Acc = 0
    num = 0
    for index, (img, target, col) in loop:
        with torch.no_grad():
            pred_r = Model.net_r(img)
            pred_g = Model.net_g(img)

            pred = pred_r * rate + pred_g * (1 - rate)
            pred = torch.argmax(pred, dim = 1)
            Acc += torch.sum(pred==target)
            num += len(target)
            # print("col:", col[0],"target:", target[0],"pred_r:", pred_r[0],"pred_g:", pred_g[0])
    return Acc/num