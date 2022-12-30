import torch
from torch.utils.data import Dataset, DataLoader


class ContrastiveDataset(Dataset):
    def __init__(self, data, model, device):
        super(ContrastiveDataset, self).__init__()
        self.data = data
        self.groups = [[], [], [], []]  # 0 for true negative, 1 for false negative, 2 for false positive, 3 for true positive

        bs = 64
        loader = DataLoader(data, batch_size = bs, shuffle = False)
        for i, (img, target, col) in enumerate(loader):
            ids = range(i * bs, min(i * bs + bs, len(data)))
            with torch.no_grad():
                img, target, col = img.to(device), target.to(device), col.to(device)
                pred, _ = model(img, col, target, False)
                pred = torch.argmax(pred, dim=1)
                for idx, p, t in zip(ids, pred, target):
                    self.groups[p * 2 + t].append(idx)

        self.group_len = [len(self.groups[i]) for i in range(4)]
        self.output_size = 4

    def __len__(self):
        return self.group_len[0] + self.group_len[3] - 2 * self.output_size + 2

    def __getitem__(self, idx):
        if idx < self.group_len[0] - self.output_size + 1:
            group = 0
            group_positive = 2
            group_negative = 1
            positive = 0
        else:
            idx = idx - self.group_len[0] + self.output_size - 1
            group = 3
            group_positive = 1
            group_negative = 2
            positive = 1

        anchors, positives, negatives = [], [], []
        for i in range(idx, idx + 4):
            # get points just use the image
            anchors.append(self.data[self.groups[group][i % self.group_len[group]]][0])
            positives.append(self.data[self.groups[group_positive][i % self.group_len[group_positive]]][0])
            negatives.append(self.data[self.groups[group_negative][i % self.group_len[group_negative]]][0])

        target_anchors = torch.tensor([positive] * 4)
        target_positives = torch.tensor([positive] * 4)
        target_negatives = torch.tensor([1 - positive] * 4)

        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives), target_anchors, target_positives, target_negatives



