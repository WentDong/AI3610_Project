import torch
from tqdm import *
from model.LeNet import MyModel
from dataset.ColorMNISTLoader import ColoredMNIST
from torch.utils.data import DataLoader
from model.BackdoorAdjustment import BackDoorAdjust
def eval(Model, TestLoader, Prior_list):
    '''
    Model: Model should have "eval", input is img, col, output is the pred_list: [P(y|x,e)) for every e]
    Prior_list: the prior probability of P(e).
    '''
    loop = tqdm(enumerate(TestLoader), total=len(TestLoader))
    Acc = 0
    num = 0
    for index, (img, target, col) in loop:
        with torch.no_grad():
            pred_list = Model.eval(img, col)
            pred = BackDoorAdjust(pred_list, Prior_list)
            pred = torch.argmax(pred, dim = 1)
            Acc += torch.sum(pred==target)
            num += len(target)
    return Acc/num


if __name__=="__main__":

    Channel = 1
    if Channel == 1:
        TestDataset = ColoredMNIST(root = '.', env='test', merge_col = True)
    else:
        TestDataset = ColoredMNIST(root = '.', env='test', merge_col = False)

    TestLoader = DataLoader(TestDataset, 8)
    Model = MyModel(Channel)
    Model = torch.load("./out/epoch0.pth")
    print(eval(Model, TestLoader, [0.5,0.5]))