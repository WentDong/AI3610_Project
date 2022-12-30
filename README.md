# AI3610_Project
Final Project of Course AI3610

## Dataset
This is a project on classifying handwritten digits with the ColoredMNIST dataset. The dataset is a modified version of the MNIST dataset, where each image is colored either red or green. The spurious feature color could mislead the network to make wrong decisions while domain changes (**OOD**). The goal is to train a model that can classify the digits whatever color the image is in.

## Methods
All our methods are for better robustness against spurious features. We will mark the models with the following tags:
- K: the method knows the spurious feature is color
    - KC: the method knows the spurious feature is color, and further, the specific color of each sample
    - DM: the method operates the dataset on the spurious feature
- UNK: the method does not know the spurious feature is color
- ED: extra data other than the train set of ColoredMNIST is used

### CLIP Classifier [K/UNK, ED]
Rather than causal inference based methods, we first tried pretrained models. They solve the OOD problem through bruteforce huge datasets. With [CLIP](https://arxiv.org/abs/2103.00020) and some class label text `"A photo of colored handwritten number [num]"`, we are able to achieve "semantic" understanding of the images. 

Run this to see the result:
```bash
python train.py --model clip --test  # do not forget to test only
```

### Dual Network [K-KC]
We use two networks, one trained with red images and the other with green images. All training images are directly from the dataset. At test time, we utilize backdoor adjustment to get a prediction.

Run this to train a dual network:
```bash
python train.py --backdoor_adjustment
```

### Dual Network with Color Augmentation [K-DM]
Further from original numbers, we use color augmentation to train the dual network. For each input image, we send a green version of it to the green network and a red version of it to the red network. At test time, we also change the input color and utilize backdoor adjustment to get a prediction.

Run this to train a dual network with color augmentation:
```bash
python train.py --backdoor_adjustment --change_col
```

### CorrectNContrast [UNK]
We follow the idea of [CorrectNContrast](https://arxiv.org/abs/2203.01517) to train a model. We first train a (single) model with the original dataset. Then we use the model to predict labels. We use the predicted labels to generate a new dataset. In the new dataset, we use the true positives and true negatives as anchors, while the false positives and false negatives as some reference samples. With these, we apply contrastive learning to train a new model.

Run this to train a CorrectNContrast model:
```bash
python train_cnc.py
```

### More Controllable Arguments
Please check out `utils/args.py`.