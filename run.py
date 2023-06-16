import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from model3 import ResNet, ResidualBlock
import tqdm
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()])

class ImageDataset(Dataset):
    """ Image shape: 32x32x3 """

    def __init__(self, root_dir, transform=None, fmt=':04d', extension='.jpg'):
        self.root_dir = root_dir
        self.fmtstr = '{' + fmt + '}' + extension
        self.transform = transform


    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.fmtstr.format(idx)
        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        data = self.transform(img)
        return data

def inference2(args, test_loader, model):
    """ model inference """

    model.eval()
    preds = []
    
    with torch.no_grad():
        pbar = tqdm.tqdm(test_loader)
        for i, x in enumerate(pbar):
            
            image = x.to(args.device)
            
            y_hat = model(image)
            
            y_hat.argmax()
        
            _, predicted = torch.max(y_hat, 1)
            preds.extend(map(lambda t: t.item(), predicted))

    return preds


if __name__ == '__main__': #import나 위에서선언한것들이 자동으로 실행되지 않고 함수가 실행이 되었을때만 시작되는거
    parser = argparse.ArgumentParser(description='2023 DL Term Project #1')
    parser.add_argument('--load-model', default='checkpoints/maa2.ckpt', help="Model's state_dict")
    parser.add_argument('--dataset', default='test_images/', help='image dataset directory') #이름을 dataset이라고 하고 생성되는 거가 default, help가 설명
    #parser.add_argument('--dataset_train', default='train_images/', help='image dataset directory_train')
    parser.add_argument('--batch-size', default=16, help='test loader batch size')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # instantiate model
    net = ResNet(ResidualBlock, [2, 2, 2]).to(device)
    #net = Net()
    net.load_state_dict(torch.load(args.load_model))

    # load dataset in test image folder
    test_data = ImageDataset(args.dataset, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)
    
    
    preds = inference2(args, test_loader, net)
    
    with open('result.txt', 'w') as f:
        f.writelines('\n'.join(map(str, preds)))