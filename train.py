import os
import torch
from torch.cuda import random
import torchvision.transforms as transforms
from torchvision.utils import save_image

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import random

from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import AlbumAudioDataset
from models import VQ_VAE, AudioEncoder

import logging

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# # Logger
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# formatter = logging.Formatter(u'%(asctime)s %(message)s')
# file_handler = logging.FileHandler(os.path.join('./results/log', '{}.log'.format(option_name)))
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# Load Data
batch_size = 16
data = pd.read_csv('./MuMu_dataset/rearranged_MuMu_dataset.csv')
total_num = len(data)

train_dataset = AlbumAudioDataset(data=data[0:int(0.97*total_num)], sr=16000, input_length=4)
test_dataset = AlbumAudioDataset(data=data[int(0.97*total_num):], sr=16000, input_length=4)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
print('Total: {} ({}/{})'.format(total_num, len(train_dataset), len(test_dataset)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sample_dir = os.path.join('./results/sample_images', 'debug')
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# Set Params TODO: 파라미터 지저분한 거 config로 빼기
num_training_updates = 10000

num_hiddens = 256
num_residual_hiddens = 256
num_residual_layers = 2

embedding_dim = 32*32*1
num_embeddings = 512

commitment_cost = 0.25
decay = 0.99
learning_rate = 1e-4

# Set Models
vq_vae = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim, commitment_cost, decay).to(device)
vq_vae.load_state_dict(torch.load('results/model/all_256_state_dict.pt')).to(device)
audio_encoder = AudioEncoder().to(device)

vq_vae.eval()
audio_encoder.train()

optimizer = torch.optim.Adam(audio_encoder.parameters(), lr=learning_rate)
train_loss= []
test_loss = []

# Train
for i in range(num_training_updates):
    album_image, track_audio = next(iter(train_loader))
    album_image = album_image.to(device)
    track_audio = track_audio.to(device)
    optimizer.zero_grad()

    # Get Album Image Latent z
    encoder_out = vq_vae._pre_vq_conv(vq_vae._encoder(album_image))
    _, image_z, _, _ = vq_vae._vq_vae(encoder_out)

    # Get Track Audio Latent z
    audio_z = audio_encoder(track_audio)

    loss = F.mse_loss(image_z, audio_z)
    loss.backward()
    optimizer.step()
    
    train_loss.append(loss.item())

    print(loss)
    break

    if (i+1) % 1000 == 0:
           
        with torch.no_grad():
            for album_image, track_audio in test_loader:
                album_image = album_image.to(device)
                track_audio = track_audio.to(device)

                # Get Album Image Latent z
                encoder_out = vq_vae._pre_vq_conv(vq_vae._encoder(album_image))
                _, image_z, _, _ = vq_vae._vq_vae(encoder_out)

                # Get Track Audio Latent z
                audio_z = audio_encoder(track_audio)

                loss = F.mse_loss(image_z, audio_z)
                test_loss.append(loss.item())
            
            # Save sample images
            reconstruction = vq_vae._decoder(audio_z)
            x_concat = torch.cat([img.view(-1, 3, 128, 128), reconstruction.view(-1, 3, 128, 128)], dim=3)
            save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(i+1)))

            # Save model
            torch.save(model, './results/audio_encoder/{}.pt'.format(i+1))

             # Logging
            logger.info('[STEP %d] train_loss: %.3f, test_loss: %.3f' % (i+1, np.mean(train_loss), np.mean(test_loss)))
            train_loss= []
            test_loss = []