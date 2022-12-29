from tqdm import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataset.ColorMNISTLoader import ColoredMNIST
import torch
from torch import nn
from model.LeNet import MyModel
from eval import eval

def train_epoch(TrainLoader, Model, Optimizer, epoch, Loss, Writer):
    loop = tqdm(enumerate(TrainLoader), total=len(TrainLoader))
    for index, (img, target, col) in loop:
        pred_r, pred_g, target_r, target_g = Model(img, col, target)
        Optimizer.zero_grad()
        loss = Loss(pred_r, target_r) * len(pred_r) + Loss(pred_g, target_g) * len(pred_g)
        loss.backward()
        Optimizer.step()
        Writer.add_scalar('train/loss', scalar_value=loss, global_step=index + epoch * len(TrainLoader))
        loop.set_description(f'In Epoch {epoch}')
        loop.set_postfix(loss = loss)


if __name__=="__main__":
    Channel = 1
    if Channel == 1:
        TrainDataset = ColoredMNIST(root = '.', env = "all_train", merge_col = True)
        TestDataset = ColoredMNIST(root = '.', env='test', merge_col = True)
    else:
        TrainDataset = ColoredMNIST(root = '.', env = "all_train", merge_col = False)
        TestDataset = ColoredMNIST(root = '.', env='test', merge_col = False)
    TrainLoader = DataLoader(TrainDataset, batch_size=32, shuffle = True)
    TestLoader = DataLoader(TestDataset, batch_size=32, shuffle = False)
    Rate = TrainDataset.num[0] / (TrainDataset.num[0]+TrainDataset.num[1]) #P(E)
    print(Rate)
    n_epoch = 1
    Loss_Function = nn.CrossEntropyLoss()
    Model = MyModel(input_channel=Channel)
    Optimizer = torch.optim.Adam(Model.parameters(), lr = 1e-5)
    Writer = SummaryWriter()

    for epoch in range(n_epoch):
        train_epoch(TrainLoader, Model, Optimizer, epoch, Loss_Function, Writer)
        Acc = eval(Model, TestLoader, Rate)
        print(f"After epoch {epoch}, the accuracy is {Acc}")
        torch.save(Model, f"./out/epoch{epoch}.pth")