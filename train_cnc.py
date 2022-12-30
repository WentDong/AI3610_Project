from tqdm import *
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.ColorMNISTLoader import ColoredMNIST
from dataset.PairDataset import PairDataset
import torch
from torch import nn
from eval import eval
import argparse
import os
from utils.args import *


def train_epoch(trainLoader, model, device, optimizer, epoch, losses, writer, change_col, backdoor_adjustment):
    loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
    for index, (img, target, col) in loop:
        img, target, col = img.to(device), target.to(device), col.to(device)

        if backdoor_adjustment:
            pred_r, pred_g, target_r, target_g = model(img, col, target, change_col)
            acc_r, acc_g = (pred_r.argmax(dim=1) == target_r).float().mean(), (pred_g.argmax(dim=1) == target_g).float().mean()
            optimizer.zero_grad()
            # loss = loss_r(pred_r, target_r) * len(pred_r) + loss_g(pred_g, target_g) * len(pred_g)
            loss = losses['r'](pred_r, target_r) + losses['g'](pred_g, target_g)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', scalar_value=loss, global_step=index + epoch * len(trainLoader))
            loop.set_description(f'In Epoch {epoch}')
            loop.set_postfix(loss=loss, acc_r=acc_r, acc_g=acc_g)
        else:
            pred, target = model(img, col, target, change_col)
            acc = (pred.argmax(dim=1) == target).float().mean()
            optimizer.zero_grad()
            loss = losses['default'](pred, target)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', scalar_value=loss, global_step=index + epoch * len(trainLoader))
            loop.set_description(f'In Epoch {epoch}')
            loop.set_postfix(loss=loss, acc=acc)


if __name__ == "__main__":
    args = get_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model.lower() == "lenet":
        from model.LeNet import MyModel as ModelClass
        Model = lambda **kwargs: ModelClass(input_channel=args.channel, backdoor_adjustment=args.backdoor_adjustment, **kwargs)
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

    loss_functions = {}
    if args.backdoor_adjustment:
        if (args.Reweight):
            loss_functions['r'] = nn.CrossEntropyLoss(
                weight=torch.tensor([trainDataset.col_label[0][1], trainDataset.col_label[0][0]]).float())
            loss_functions['g'] = nn.CrossEntropyLoss(
                weight=torch.tensor([trainDataset.col_label[1][1], trainDataset.col_label[1][0]]).float())
        else:
            loss_functions['r'] = nn.CrossEntropyLoss()
            loss_functions['g'] = nn.CrossEntropyLoss()
    else:
        loss_functions['default'] = nn.CrossEntropyLoss()

    model = Model(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    if args.train:
        # ERM
        ckpt = f"./out/latest_channel{channel}.pth"
        if os.path.exists(ckpt):
            try:
                model.load_state_dict(torch.load(ckpt, map_location=device))
            except TypeError:
                retrain = True
            else:
                retrain = True
        else:
            retrain = True

        if retrain:
            for epoch in range(args.n_epoch):
                train_epoch(trainLoader, model, device, optimizer, epoch, loss_functions, writer,
                            args.change_col, args.backdoor_adjustment)
                if args.backdoor_adjustment:
                    acc = eval(model, device, testLoader, [rate, 1 - rate], args.change_col)
                else:
                    acc = eval(model, device, testLoader, [1.], args.change_col)
                print(f"After epoch {epoch}, the accuracy is {acc}")
                torch.save(model.state_dict(), f"./out/epoch{epoch}_channel{channel}.pth")
                torch.save(model.state_dict(), f"./out/latest_channel{channel}.pth")

        # CNC
        PairDataset(trainDataset, model, device)
        for epoch in range(args.n_epoch_cnc):
            train_epoch(trainLoader, model, device, optimizer, epoch, loss_function_r, loss_function_g, writer, args.change_col)


    else:
        if args.backdoor_adjustment:
            acc = eval(model, device, testLoader, [rate, 1 - rate], args.change_col)
        else:
            acc = eval(model, device, testLoader, [1.], args.change_col)
        print(f"The accuracy is {acc}")

