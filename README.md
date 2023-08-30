# Deep_learning_cifar10_dataset

# PyTorch models trained on CIFAR-10 dataset

I modified TorchVision official implementation of popular CNN models, and trained those on CIFAR-10 dataset.
I changed number of class, filter size, stride, and padding in the the original code so that it works with CIFAR-10.
I also share the weights of these models, so you can just load the weights and use them.
The code is highly re-producible and readable by using PyTorch-Lightning

# training of models, accuracy
1. CNN -> Use Vscode
   
![image](https://github.com/jeongdaeyun/Deep_learning_cifar10_dataset/assets/50974241/43ab4fcc-2bfd-4967-a9d4-ff4f8c5d66cb)

2. CNN에서 Resnet의 resudial block 개념을 이용 -> Use Colab

![image](https://github.com/jeongdaeyun/Deep_learning_cifar10_dataset/assets/50974241/bc505980-e5cf-42b5-bfa7-5198c1f51e5c)


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
