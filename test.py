import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model import DLA
import tqdm
from PIL import Image

dir_path = "./test_images/0000.jpg"

img = Image.open(dir_path)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
img = transform(img)

checkpoint_path = './checkpoints/ckpt.pth'
model = DLA()
model.load_state_dict(torch.load(checkpoint_path["net"]))
model.eval()

with torch.no_grad():
    output = model(img.unsqueeze(0))
preds = torch.argmax(output, dim=1)

class_names = ['0', '1', '2','3','4','5','6','7','8','9'] # 분류할 클래스 이름
pred_class = class_names[preds.item()]

print(pred_class)