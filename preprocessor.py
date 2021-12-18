import os
import csv
import pickle
import random
import torch
import torchvision.transforms as transforms
import torchaudio
import numpy as np
import pandas as pd
import PIL.Image as Image

from short_chunk_CNN import ShortChunkCNN_Res

class Preprocessor:
    def __init__(self, npy_path, sr=16000, start_sec=5, duration_sec=4):
        self.sr = sr
        self.duration_sec = duration_sec
        self.start_sec = start_sec
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Paths control
        self.img_path = os.path.join(npy_path, 'image_latents')
        self.audio_path = os.path.join(npy_path, 'audio_embs')
        if not os.path.exists(npy_path):
            os.makedirs(self.img_path)
            os.makedirs(self.audio_path)

        # MSD path setting
        self.MSD_path = "../../../media/bach2/dataset/MSD/songs"
        self.id_to_path = pickle.load(open("./MuMu_dataset/7D_id_to_path.pkl",'rb'))
        self.MSD_id_to_7D_id = pickle.load(open("./MuMu_dataset/MSD_id_to_7D_id.pkl",'rb'))

        # Load pretrained audio feature extractor
        pretrained_weights = torch.load("./sota-music-tagging-models/models/msd/short_res/best_model.pth")
        self.audio_feature_extractor = ShortChunkCNN_Res().to(self.device)
        self.audio_feature_extractor.load_state_dict(pretrained_weights, strict=False)
        self.audio_feature_extractor.eval()

        # Load VQ-VAE
        self.trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        self.vq_vae = torch.load('./results/model/all_256.pt').to(self.device)
        self.vq_vae.eval()
    
    ### Utils
    def get_MSD_path(self, MSD_id):
        return os.path.join(self.MSD_path, self.id_to_path[self.MSD_id_to_7D_id[MSD_id]])
        
    def get_chunk_waveform(self, MSD_id):
        f_path = self.get_MSD_path(MSD_id)
        waveform, sample_rate = torchaudio.load(f_path) 
        transformed = torchaudio.transforms.Resample(sample_rate, self.sr)(waveform[0,:].view(1,-1))
        chunk_start = self.sr*self.start_sec
        duration = self.sr*self.duration_sec
        transformed = transformed[:,chunk_start:chunk_start+duration]
        return transformed

    def get_audio_emb_path(self, MSD_id):
        return os.path.join(self.audio_path, '{}.npy'.format(MSD_id))

    def get_img_latent_path(self, img_path):
        f_name = (os.path.split(img_path)[-1]).split('.')[0]
        return os.path.join(self.img_path, '{}.npy'.format(f_name))
        
    ### Preprocessing
    def save_audio_emb(self, MSD_id):
        waveform = self.get_chunk_waveform(MSD_id)
        waveform = waveform.unsqueeze(0).to(self.device)
        emb = self.audio_feature_extractor(waveform)
        emb = emb.detach().cpu().numpy()
        emb_path = self.get_audio_emb_path(MSD_id)
        np.save(emb_path, emb)
        return emb_path
    
    def save_image_latent(self, img_path):
        image = self.trans(Image.open(img_path).convert('RGB'))
        image = image.unsqueeze(0).to(self.device)
        encoder_out = self.vq_vae._pre_vq_conv(self.vq_vae._encoder(image))
        _, latent, _, _ = self.vq_vae._vq_vae(encoder_out)
        latent = latent.squeeze(0).detach().cpu().numpy()
        latent_path = self.get_img_latent_path(img_path)
        np.save(latent_path, latent)
        return latent_path

    def preprocessing(self, data):
        """
        data: pandas.DataFrame, should be from 'rearranged_MuMu_dataset.csv'
        """
        total_num = len(data)
        print("Start download {} num of data".format(total_num))

        for idx in range(total_num):
            if (idx+1) % 10000 == 0:
                print("{}% Downloaded...".format(int(idx/total_num*100)))

            track = data.iloc[idx]

            img_path = track['album_img_path']
            latent_path = self.save_image_latent(img_path)
            track['album_img_path'] = latent_path

            MSD_id = track['MSD_track_id']
            emb_path = self.save_audio_emb(MSD_id)
            track['MSD_track_id'] = emb_path

        return data

    ### Sample (Use After Preprocessing)
    def sample_random(self, data):
        """
        data: pandas.DataFrame, should be from 'rearranged_MuMu_dataset.csv'
        """
        track = data.iloc[random.randint(0, len(data)-1)]
        MSD_id = track['MSD_track_id']
        img_path = track['album_img_path']

        img = self.trans(Image.open(img_path).convert("RGB"))
        img_latent_z = np.load(self.get_img_latent_path(img_path))

        audio = self.get_chunk_waveform(MSD_id)
        audio_emb = np.load(self.get_audio_emb_path(MSD_id))

        return (img, img_latent_z), (audio, audio_emb)