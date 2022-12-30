import torch
from torch.nn import Module


class ContrastLoss(Module):
    def __init__(self, temperature=0.1):
        super(ContrastLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feat):
        anchors, positives, negatives = feat['anchor'], feat['positive'], feat['negative']
        anchor = anchors[:, 0]
        loss_1 = self.loss(anchor, positives, negatives)
        positive = positives[:, 0]
        loss_2 = self.loss(positive, anchors, negatives)
        return loss_1 + loss_2

    def loss(self, anchor, positives, negatives):
        pos = torch.exp(anchor @ positives / self.temperature)
        neg = torch.exp(anchor @ negatives / self.temperature)
        dominator = pos.sum(dim=1) + neg.sum(dim=1)
        loss = - torch.log(pos / dominator).mean()
        return loss