import time
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
import torch.optim as optim

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import Dataset

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

from gcn_lib.torch_vertex import Grapher, act_layer

class Decoder(nn.Module):
    def __init__(self, num_channels=3):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)

class Image2Graph(nn.Module):

    def __init__(self):
        self.inplanes = 64
        super(Image2Graph, self).__init__()
        
        self.model = models.resnet18(pretrained=False)
        self.model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        self.localG = Grapher(in_channels=256, kernel_size=21, dilation=1,
                              act='relu', norm="batch", bias=True, stochastic=False,
                              epsilon=0.0, r=1, n=14 * 14, drop_path=0.0, relative_pos=True, isVideo=False)
        self.localG2 = Grapher(in_channels=512, kernel_size=21, dilation=1,
                               act='relu', norm="batch", bias=True, stochastic=False,
                               epsilon=0.0, r=1, n=7 * 7, drop_path=0.0, relative_pos=True, isVideo=False)
        
        self.alpha = nn.Parameter(torch.ones(4), requires_grad=True)
        self.decoder = Decoder(num_channels=3)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1)
 
    def forward(self, x):
        N, C, H, W = x.size()
        x_raw = x
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # torch.Size([N, C=64, T, H=56, W=56])
        x = self.model.layer1(x)  # ([N, C=64, T, H=56, W=56])
        x = self.model.layer2(x)  # ([N, C=128, T, H=28, W=28])
        x = self.model.layer3(x)  # ([N, C=256, T, H=14, W=14])
        
        N, C, H, W = x.size()
        x = x + self.localG(x,N) * self.alpha[0]
        x = x.view(N, C, H, W)

        x = self.model.layer4(x)  # ([N, C=512, T, H=7, W=7])
        N, C, H, W = x.size()
        x = x + self.localG2(x,N) * self.alpha[1]
        
        x_restored = self.decoder(x)
        ssim = self.ssim(torch.sigmoid(x_raw), x_restored)
        return x_restored, 1-ssim

    def loss(self, x_raw, x_restored):
        return 1-self.ssim(torch.sigmoid(x_raw), x_restored)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_epochs = 300
batch_size = 32
learning_rate = 0.001
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_filenames = sorted([f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

train_dataset = ImageDataset(directory='imageDataset/04July_2011_Monday_heute_default-0/1', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)

model = Image2Graph()
model.to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {num_params}")
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

import time

# Training loop
start_time = time.time()
for epoch in range(num_epochs):
    image_id = 0
    model.train()
    epoch_loss = 0.0
    for images in train_loader:
        images = images.to(device)
        outputs, loss = model(images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
end_time = time.time()

print(f"training time for {num_epochs} epoch {end_time-start_time}")
print(f"average time per epoch {(end_time-start_time)/num_epochs}")
print("Training complete!")