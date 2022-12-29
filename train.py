from tqdm import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.ColorMNISTLoader import ColoredMNIST
import torch
from torch import nn
from eval import eval
import argparse
import os
from utils.args import *
def train_epoch(TrainLoader, Model, Optimizer, epoch, Loss_r, Loss_g, Writer):
    loop = tqdm(enumerate(TrainLoader), total=len(TrainLoader))
    for index, (img, target, col) in loop:
        pred_r, pred_g, target_r, target_g = Model(img, col, target)
        Optimizer.zero_grad()
        loss = Loss_r(pred_r, target_r) * len(pred_r) + Loss_g(pred_g, target_g) * len(pred_g)
        loss.backward()
        Optimizer.step()
        Writer.add_scalar('train/loss', scalar_value=loss, global_step=index + epoch * len(TrainLoader))
        loop.set_description(f'In Epoch {epoch}')
        loop.set_postfix(loss = loss)

if __name__=="__main__":
    args = get_args()
    channel = args.channel
    if channel == 1:
        trainDataset = ColoredMNIST(root = args.root_path, env = "all_train", merge_col = True)
        testDataset = ColoredMNIST(root = args.root_path, env='test', merge_col = True)
    else:
        trainDataset = ColoredMNIST(root = args.root_path, env = "all_train", merge_col = False)
        testDataset = ColoredMNIST(root = args.root_path, env='test', merge_col = False)
    trainLoader = DataLoader(trainDataset, batch_size=args.bs, shuffle = True)
    testLoader = DataLoader(testDataset, batch_size=args.bs, shuffle = False)

    rate = trainDataset.num[0] / (trainDataset.num[0]+trainDataset.num[1]) #P(E)
    print(rate)
    n_epoch = 3
    print(trainDataset.col_label)
    loss_function_r = nn.CrossEntropyLoss(weight=torch.tensor([trainDataset.col_label[0][1], trainDataset.col_label[0][0]]).float())
    loss_function_g = nn.CrossEntropyLoss(weight=torch.tensor([trainDataset.col_label[1][1], trainDataset.col_label[1][0]]).float())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model.lower() == "lenet":
        from model.LeNet import MyModel as Model
    elif args.model.lower() == "clip":
        from model.CLIP import CLIPClassifier as Model
    model = Model(input_channel=channel, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    for epoch in range(n_epoch):
        train_epoch(trainLoader, model, optimizer, epoch, loss_function_r, loss_function_g, writer)
        if args.backdoor_adjustment:
            acc = eval(model, testLoader, [rate, 1-rate])
        else:
            acc = eval(model, testLoader, [1.])
        print(f"After epoch {epoch}, the accuracy is {acc}")
        torch.save(model, f"./out/epoch{epoch}_channel{channel}.pth" )
