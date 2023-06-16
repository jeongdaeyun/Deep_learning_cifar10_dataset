# Deep_learning_cifar10_dataset

# PyTorch models trained on CIFAR-10 dataset

I modified TorchVision official implementation of popular CNN models, and trained those on CIFAR-10 dataset.
I changed number of class, filter size, stride, and padding in the the original code so that it works with CIFAR-10.
I also share the weights of these models, so you can just load the weights and use them.
The code is highly re-producible and readable by using PyTorch-Lightning

# Statistics of supported models
![image](https://github.com/jeongdaeyun/Deep_learning_cifar10_dataset/assets/50974241/eb1c1b11-3f0d-494a-ae9a-c91d2c8abf63)

# Dataset download
1. you import library
  import ssl
  import torch
  import torch.nn as nn
  from torchvision import transforms, datasets
2. Download image -> only train, you must download test, too.
  train_dataset = datasets.CIFAR10(root="./data/",
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor()
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
                                           


# How to test this project
1. git clone 
2. The model is built. -> I referred to resnet as the model
 -> The reason why I chose resnet is
  a. vanishing gradient
  b. resnet is good model
3. python run.py


# Requirements
Just to use pretrained models
  pytorch = 1.7.0
