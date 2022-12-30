import torch
from torch import nn


class LeNet(nn.Module):
    def __init__(self, input_channel=1, output_channel=2):
        super(LeNet, self).__init__()

        self.extract = nn.Sequential(
            nn.Conv2d(input_channel, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
        )
        self.final = nn.Linear(84, output_channel)

    def forward(self, x):
        feat = self.extract(x)
        pred = self.final(feat)
        return pred, feat

    def feat(self, x):
        return self.extract(x)

    def pred(self, x):
        x = self.extract(x)
        return self.final(x)


class CorrectNContrast(nn.Module):
    def __init__(self, input_channel=1, output_channel=2, device='cuda'):
        super(CorrectNContrast, self).__init__()
        self.device = device
        self.net = LeNet(input_channel, output_channel).to(device)

    def forward(self, anchors, positives, negatives, target_anchors, target_positives, target_negatives):
        anchors_flat = anchors.view(-1, *anchors.shape[-3:])
        positives_flat = positives.view(-1, *positives.shape[-3:])
        negatives_flat = negatives.view(-1, *negatives.shape[-3:])
        imgs_all = torch.cat([anchors_flat, positives_flat, negatives_flat], dim=0)
        target_all = torch.cat([target_anchors, target_positives, target_negatives], dim=0).view(-1)

        pred_all = self.net.pred(imgs_all)
        feat_all = self.net.feat(imgs_all)

        feat = {
            'anchor': feat_all[:len(anchors_flat)].reshape(*anchors.shape[:2], -1),
            'positive': feat_all[len(anchors_flat):len(anchors_flat) + len(positives_flat)].reshape(*positives.shape[:2], -1),
            'negative': feat_all[len(anchors_flat) + len(positives_flat):].reshape(*negatives.shape[:2], -1),
        }

        return pred_all, target_all, feat

    def eval(self, x, col, change_col=False):
        return [self.net.pred(x)]
