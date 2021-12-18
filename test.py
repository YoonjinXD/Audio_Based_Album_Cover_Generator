import os
import torch
from torchvision.utils import save_image
import torchaudio

import pandas as pd
import numpy as np

from models import VQ_VAE, Audio2ImageNet
from preprocessor import Preprocessor

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Test sample number
test_num = 5

# Set Path Dir
image_sample_dir = os.path.join('./results', 'images')
audio_sample_dir = os.path.join('./results', 'audios')
if not os.path.exists('./results'):
    os.makedirs(image_sample_dir)
    os.makedirs(audio_sample_dir)

# Load Data & Preprocessor
data = pd.read_csv('./MuMu_dataset/rearranged_MuMu_dataset.csv')
preprocessor = Preprocessor('./preprocessed_data')

# Set Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vq_vae = torch.load('results/model/all_256.pt').to(device)
audio2image_net = torch.load('results/model/step-30.pt').to(device) # change testing model by step
audio2image_net.eval()

# Make Samples
for i in range(test_num):
    (origin_img, origin_z), (audio, audio_emb) = preprocessor.sample_random(data)
    audio_emb = torch.tensor(audio_emb).to(device)
    audio_emb = audio_emb.unsqueeze(0)
    audio_z = audio2image_net(audio_emb)
    _, quantized, _, _ = vq_vae._vq_vae(audio_z)

    # Check the quantized vector difference from original q
    # origin_z = torch.tensor(origin_z).to(device)
    # origin_z = origin_z.unsqueeze(0)
    # _, origin_quantized, _, _ = vq_vae._vq_vae(origin_z)
    # print(torch.mean(quantized), torch.mean(origin_quantized))
    
    # Save last generated test sample
    generated_img = vq_vae._decoder(quantized)
    generated_img = generated_img.squeeze(0).detach().cpu()
    concat = torch.cat([origin_img.view(-1, 128, 128), generated_img.view(-1, 128, 128)], dim=2)
    save_image(concat, os.path.join(image_sample_dir, 'test-{}.png'.format(i+1)))

    # Save input audio samples
    torchaudio.save(os.path.join(audio_sample_dir, 'test-{}.wav'.format(i+1)), src=audio, sample_rate=16000)

print("Samples Created")

