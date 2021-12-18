import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import TripletDataset
from models import Audio2ImageNet

import logging

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Loss
class TripletLoss(nn.Module):
    def __init__(self, margin):
        """
        Args:
          margin:
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()
    
    def forward(self, anchor, positive, negative):
        pos_sim = nn.CosineSimilarity(dim=-1)(anchor, positive)
        neg_sim = nn.CosineSimilarity(dim=-1)(anchor, negative)
        losses = self.relu(self.margin - pos_sim + neg_sim)
        return losses.mean()

# Logger
log_file_name = 'train_audio_encoder'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s %(message)s')
file_handler = logging.FileHandler(os.path.join('./results/log', '{}.log'.format(log_file_name)))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load Data
batch_size = 16
data = pd.read_csv('./MuMu_dataset/npy_list.csv')
total_num = len(data)

train_dataset = TripletDataset(data=data[0:int(0.97*total_num)])
test_dataset = TripletDataset(data=data[int(0.97*total_num):])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
print('Total: {} ({}/{})'.format(total_num, len(train_dataset), len(test_dataset)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image_sample_dir = os.path.join('./results', 'images')
audio_sample_dir = os.path.join('./results', 'audios')
if not os.path.exists('./results'):
    os.makedirs(image_sample_dir)
    os.makedirs(audio_sample_dir)

# Set Params
num_training_updates = 50
learning_rate = 1e-3

# Set Models
audio2image_net = Audio2ImageNet().to(device)
audio2image_net.train()
optimizer = torch.optim.Adam(audio2image_net.parameters(), lr=learning_rate)
criterion = TripletLoss(margin=0.4).to(device)

# Train
print("Start Training")
for i in range(num_training_updates):
    audio_emb, pos_z, neg_z = next(iter(train_loader))
    audio_emb = audio_emb.to(device)
    pos_z = pos_z.to(device)
    neg_z = neg_z.to(device)

    audio_z = audio2image_net(audio_emb)

    loss = criterion(audio_z, pos_z, neg_z)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    train_loss = loss.detach().cpu().item()

    if (i+1) % 1 == 0:
        with torch.no_grad():
            audio_emb, pos_z, neg_z = next(iter(test_loader))
            audio_emb = audio_emb.to(device)
            pos_z = pos_z.to(device)
            neg_z = neg_z.to(device)

            audio_z = audio2image_net(audio_emb)

            loss = criterion(audio_z, pos_z, neg_z)
            test_loss= loss.detach().cpu().item()

            # Save model
            torch.save(audio2image_net, './results/model/step-{}.pt'.format(i+1))

            # Logging
            logger.info('[STEP %d] train_loss: %.3f, test_loss: %.3f' % (i+1, np.mean(train_loss), np.mean(test_loss)))
            print('[STEP %d] train_loss: %.3f, test_loss: %.3f' % (i+1, np.mean(train_loss), np.mean(test_loss)))