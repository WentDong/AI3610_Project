import torch
from tqdm import *
from model.LeNet import MyModel
from dataset.ColorMNISTLoader import ColoredMNIST
from torch.utils.data import DataLoader
from model.BackdoorAdjustment import BackDoorAdjust


def eval(model, device, testLoader, prior_list):
    '''
    Model: Model should have "eval", input is img, col, output is the pred_list: [P(y|x,e)) for every e]
    Prior_list: the prior probability of P(e).
    '''
    loop = tqdm(enumerate(testLoader), total=len(testLoader))
    acc = 0
    num = 0
    for index, (img, target, col) in loop:
        with torch.no_grad():
            img, target, col = img.to(device), target.to(device), col.to(device)
            
            pred_list = model.eval(img, col)
            pred = BackDoorAdjust(pred_list, prior_list)
            pred = torch.argmax(pred, dim = 1)
            acc += torch.sum(pred==target)
            num += len(target)
    return acc/num


if __name__=="__main__":

    Channel = 1
    if Channel == 1:
        TestDataset = ColoredMNIST(root = '.', env='test', merge_col = True)
    else:
        TestDataset = ColoredMNIST(root = '.', env='test', merge_col = False)

    testLoader = DataLoader(TestDataset, 8)
    Model = MyModel(Channel)
    Model = torch.load("./out/epoch0.pth")
    print(eval(Model, testLoader, [0.5,0.5]))