from tqdm import *
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.ColorMNISTLoader import ColoredMNIST
import torch
from torch import nn
from eval import eval
import argparse
import os
from utils.args import *


def train_epoch(trainLoader, model, device, optimizer, epoch, loss_r, loss_g, writer):
    loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
    for index, (img, target, col) in loop:
        img, target, col = img.to(device), target.to(device), col.to(device)

        pred_r, pred_g, target_r, target_g = model(img, col, target)
        optimizer.zero_grad()
        # loss = loss_r(pred_r, target_r) * len(pred_r) + loss_g(pred_g, target_g) * len(pred_g)
        loss = loss_r(pred_r, target_r) + loss_g(pred_g, target_g)
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', scalar_value=loss, global_step=index + epoch * len(trainLoader))
        loop.set_description(f'In Epoch {epoch}')
        loop.set_postfix(loss=loss)


if __name__ == "__main__":
    args = get_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model.lower() == "lenet":
        from model.LeNet import MyModel as Model
        transform = None
    elif args.model.lower() == "clip":
        from model.CLIP import CLIPClassifier as Model

        transform = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(224),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    channel = args.channel
    trainDataset = ColoredMNIST(root=args.root_path, env=args.trainset, merge_col=channel == 1, transform=transform)
    testDataset = ColoredMNIST(root=args.root_path, env='test', merge_col=channel == 1, transform=transform)
    trainLoader = DataLoader(trainDataset, batch_size=args.bs, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=args.bs, shuffle=False)

    rate = trainDataset.num[0] / (trainDataset.num[0] + trainDataset.num[1])  # P(E=0)
    print(rate)
    print(trainDataset.col_label)
    if (args.Reweight):
        loss_function_r = nn.CrossEntropyLoss(
            weight=torch.tensor([trainDataset.col_label[0][1], trainDataset.col_label[0][0]]).float())
        loss_function_g = nn.CrossEntropyLoss(
            weight=torch.tensor([trainDataset.col_label[1][1], trainDataset.col_label[1][0]]).float())
    else:
        loss_function_r = nn.CrossEntropyLoss()
        loss_function_g = nn.CrossEntropyLoss()
    model = Model(input_channel=channel, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)


    if args.train:
        for epoch in range(args.n_epoch):
            train_epoch(trainLoader, model, device, optimizer, epoch, loss_function_r, loss_function_g, writer)
            if args.backdoor_adjustment:
                acc = eval(model, device, testLoader, [rate, 1 - rate])
            else:
                acc = eval(model, device, testLoader, [1.])
            print(f"After epoch {epoch}, the accuracy is {acc}")
            torch.save(model, f"./out/epoch{epoch}_channel{channel}.pth")
    else:
        if args.backdoor_adjustment:
            acc = eval(model, device, testLoader, [rate, 1 - rate])
        else:
            acc = eval(model, device, testLoader, [1.])
        print(f"The accuracy is {acc}")

