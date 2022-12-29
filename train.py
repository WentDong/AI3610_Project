from tqdm import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.ColorMNISTLoader import ColoredMNIST
import torch
from torch import nn
from model.LeNet import MyModel
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
    Channel = args.channel
    if Channel == 1:
        TrainDataset = ColoredMNIST(root = args.root_path, env = "all_train", merge_col = True)
        TestDataset = ColoredMNIST(root = args.root_path, env='test', merge_col = True)
    else:
        TrainDataset = ColoredMNIST(root = args.root_path, env = "all_train", merge_col = False)
        TestDataset = ColoredMNIST(root = args.root_path, env='test', merge_col = False)
    TrainLoader = DataLoader(TrainDataset, batch_size=args.bs, shuffle = True)
    TestLoader = DataLoader(TestDataset, batch_size=args.bs, shuffle = False)

    Rate = TrainDataset.num[0] / (TrainDataset.num[0]+TrainDataset.num[1]) #P(E)
    print(Rate)
    n_epoch = 3
    print(TrainDataset.col_label)
    Loss_Function_r = nn.CrossEntropyLoss(weight=torch.tensor([TrainDataset.col_label[0][1], TrainDataset.col_label[0][0]]).float())
    Loss_Function_g = nn.CrossEntropyLoss(weight=torch.tensor([TrainDataset.col_label[1][1], TrainDataset.col_label[1][0]]).float())

    Model = MyModel(input_channel=Channel)
    Optimizer = torch.optim.Adam(Model.parameters(), lr = args.lr)
    Writer = SummaryWriter()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    for epoch in range(n_epoch):
        train_epoch(TrainLoader, Model, Optimizer, epoch, Loss_Function_r, Loss_Function_g, Writer)
        Acc = eval(Model, TestLoader, [Rate,1-Rate])
        print(f"After epoch {epoch}, the accuracy is {Acc}")
        torch.save(Model, f"./out/epoch{epoch}_channel{Channel}.pth" )