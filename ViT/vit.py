import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
from torch.nn import Conv2d, MaxPool2d, Linear, Dropout, ReLU, BatchNorm2d, Flatten
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from PIL import Image
from vision_transformer_pytorch import VisionTransformer


model = VisionTransformer.from_pretrained('ViT-B_16')
model.eval()

img = torch.randn(1, 3, 224, 224)
preds = model(img)  # (1, 1000)
print(preds)
