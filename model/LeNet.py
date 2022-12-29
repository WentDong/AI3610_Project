import torch
from torch import nn

class LeNet(nn.Module):
    def __init__(self, input_channel=1, output_channel=2):
        super(LeNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channel, 6, kernel_size=5, padding=2), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, output_channel),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

class MyModel(nn.Module):
    def __init__(self, input_channel = 1, output_channel = 2, device = 'cuda'):
        super(MyModel, self).__init__()
        self.device = device
        self.net_r = LeNet(input_channel, output_channel).to(device)
        self.net_g = LeNet(input_channel, output_channel).to(device)

    def forward(self, x, col, target):
        mask_r = (col==0)
        mask_g = (col==1)
        # pred_r = self.net_r(x)
        # pred_g = self.net_g(x)
        pred_r = self.net_r(x[mask_r])
        pred_g = self.net_g(x[mask_g])
        # target_r = target
        # target_g = target
        target_r = target[mask_r]
        target_g = target[mask_g]
        return pred_r, pred_g, target_r, target_g
        
    def eval(self, x, col, change_col=False):
        if change_col:
            x_r = torch.zeros_like(x)
            x_r[:, 0, :, :] = x.sum(dim=1)
            x_g = torch.zeros_like(x)
            x_g[:, 1, :, :] = x.sum(dim=1)
        else:
            x_r = x
            x_g = x

        pred_r = self.net_r(x_r)
        pred_g = self.net_g(x_g)
        return [pred_r, pred_g]
        