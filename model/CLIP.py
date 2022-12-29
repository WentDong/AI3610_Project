from torch import nn
import torch
import clip

numbers = range(10)
template = "A photo of colored handwritten number [num]"
label_text = [template.replace("[num]", str(num)) for num in numbers]


class CLIPClassifier(nn.Module):
    def __init__(self, input_channel=1, output_channel=2, device='cuda'):
        super(CLIPClassifier, self).__init__()
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.label_token = clip.tokenize(label_text).to(device)

    def forward(self, x, col, target):
        logits_per_image, logits_per_text = self.model(x, self.label_token)
        pred = logits_per_image.softmax(dim=-1)
        pred = pred[:, :5].sum(dim=1, keepdims=True)
        pred = [torch.cat([pred, 1 - pred], dim=1)]
        return pred, target

    def eval(self, x, col, change_col=False):
        with torch.no_grad():
            pred = self.forward(x, col, None)[0]
        return pred

