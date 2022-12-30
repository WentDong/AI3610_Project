from tqdm import *
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms.functional import InterpolationMode
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.ColorMNISTLoader import ColoredMNIST
from dataset.ContrastiveDataset import ContrastiveDataset
import torch
from torch import nn
from eval import eval
import argparse
import os
from utils.args import *
from model.CorrectNContrast import CorrectNContrast
from utils.loss import ContrastLoss


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
            loop.set_postfix(loss=loss.detach().cpu().item(), acc_r=acc_r.detach().cpu().item(), acc_g=acc_g.detach().cpu().item())
        else:
            pred, target = model(img, col, target, change_col)
            acc = (pred.argmax(dim=1) == target).float().mean()
            optimizer.zero_grad()
            loss = losses['default'](pred, target)
            loss.backward()
            optimizer.step()
            writer.add_scalar('train/loss', scalar_value=loss, global_step=index + epoch * len(trainLoader))
            loop.set_description(f'In Epoch {epoch}')
            loop.set_postfix(loss=loss.detach().cpu().item(), acc=acc.detach().cpu().item())


def train_epoch_cnc(trainLoader, model, device, optimizer, epoch, losses, writer):
    loop = tqdm(enumerate(trainLoader), total=len(trainLoader))
    for index, data in loop:
        data = [d.to(device) for d in data]

        pred, target, feat = model(*data)

        acc = (pred.argmax(dim=1) == target).float().mean()

        loss_dict = {
            'contrast': losses['contrast'](feat),
            'cross_entropy': losses['cross_entropy'](pred, target),
        }
        loss = loss_dict['contrast'] * losses['lambda'] + loss_dict['cross_entropy'] * (1 - losses['lambda'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/cnc/loss_contrast', scalar_value=loss_dict['contrast'], global_step=index + epoch * len(trainLoader))
        writer.add_scalar('train/cnc/loss_cross_entropy', scalar_value=loss_dict['cross_entropy'], global_step=index + epoch * len(trainLoader))
        writer.add_scalar('train/cnc/loss', scalar_value=loss, global_step=index + epoch * len(trainLoader))
        loop.set_description(f'In Epoch {epoch}')
        loop.set_postfix(loss=loss.detach().cpu().item(), acc=acc.detach().cpu().item())


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


    # cnc loss
    losses_cnc = {
        'contrast': ContrastLoss(args.temperature),
        'cross_entropy': nn.CrossEntropyLoss(),
        'lambda': args.lambda_contrast
    }

    model = CorrectNContrast(input_channel=channel, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    if args.train:
        model_erm = Model(device=device)
        writer = SummaryWriter()
        # ERM
        ckpt = f"./out/latest_channel{channel}.pth"
        train_erm = True
        if not args.force_train_erm and os.path.exists(ckpt):
            try:
                model_erm.load_state_dict(torch.load(ckpt, map_location=device))
                train_erm = False
            except TypeError:
                pass

        if train_erm:
            print('[INFO] Good erm checkpoint not found, start training erm ...')
            # erm loss
            loss_functions = {}
            if args.backdoor_adjustment:
                if (args.reweight):
                    loss_functions['r'] = nn.CrossEntropyLoss(
                        weight=torch.tensor([trainDataset.col_label[0][1], trainDataset.col_label[0][0]]).float())
                    loss_functions['g'] = nn.CrossEntropyLoss(
                        weight=torch.tensor([trainDataset.col_label[1][1], trainDataset.col_label[1][0]]).float())
                else:
                    loss_functions['r'] = nn.CrossEntropyLoss()
                    loss_functions['g'] = nn.CrossEntropyLoss()
            else:
                loss_functions['default'] = nn.CrossEntropyLoss()

            optimizer_erm = torch.optim.Adam(model_erm.parameters(), lr=args.lr)
            for epoch in range(args.n_epoch):
                train_epoch(trainLoader, model_erm, device, optimizer_erm, epoch, loss_functions, writer,
                            args.change_col, args.backdoor_adjustment)
                if args.backdoor_adjustment:
                    acc = eval(model_erm, device, testLoader, [rate, 1 - rate], args.change_col)
                else:
                    acc = eval(model_erm, device, testLoader, [1.], args.change_col)
                print(f"After epoch {epoch}, the accuracy is {acc}")
                torch.save(model_erm.state_dict(), f"./out/epoch{epoch}_channel{channel}.pth")
                torch.save(model_erm.state_dict(), f"./out/latest_channel{channel}.pth")

        # CNC
        print('[INFO] Start training CNC ...')
        cncDataset = ContrastiveDataset(trainDataset, model_erm, device)
        cncLoader = DataLoader(cncDataset, batch_size=args.bs, shuffle=True)
        for epoch in range(args.n_epoch_cnc):
            train_epoch_cnc(cncLoader, model, device, optimizer, epoch, losses_cnc, writer)
            acc = eval(model, device, testLoader, [1.])
            print(f"After epoch {epoch}, the accuracy is {acc}")
            torch.save(model.state_dict(), f"./out/cnc_epoch{epoch}_channel{channel}.pth")
            torch.save(model.state_dict(), f"./out/cnc_latest_channel{channel}.pth")
    else:
        acc = eval(model, device, testLoader, [1.])
        print(f"The accuracy is {acc}")

