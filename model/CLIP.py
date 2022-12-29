from torch import nn
import torch
import clip

numbers = range(10)
label_text = [f"A colored handwritten number {n}." for n in numbers]
label_token = [clip.tokenize(text) for text in label_text]

class CLIPClassifier(nn.Module):
    def __init__(self, input_channel=1, output_channel=2, device='cuda'):
        super(CLIPClassifier, self).__init__()
        model, preprocess = clip.load("ViT-B/32", device=device)

    def forward(self, x, col, target):

        image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)