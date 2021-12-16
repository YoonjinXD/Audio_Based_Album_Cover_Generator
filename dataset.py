from numpy.lib.type_check import _imag_dispatcher
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as Image
from sklearn.preprocessing import MultiLabelBinarizer

import pickle
import os
import torchaudio
import torch

class AlbumAudioDataset(Dataset):
    def __init__(self, data, sr, input_length):
        super().__init__()
        self.data = data
        self.sr = sr
        self.input_length = input_length

        self.MSD_path = "../../../media/bach2/dataset/MSD/songs"
        self.id_to_path = pickle.load(open("./MuMu_dataset/7D_id_to_path.pkl",'rb'))
        self.MSD_id_to_7D_id = pickle.load(open("./MuMu_dataset/MSD_id_to_7D_id.pkl",'rb'))

        self.trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

    def __getitem__(self, index):
        # Album Image
        track = self.data.iloc[index]
        img_path = track['album_img_path']
        img = self.trans(Image.open(img_path).convert("RGB"))

        # Track Audio
        MSD_id = track['MSD_track_id']
        audio_path = os.path.join(self.MSD_path, self.id_to_path[self.MSD_id_to_7D_id[MSD_id]])
        waveform, sample_rate = torchaudio.load(audio_path) 
        transformed = torchaudio.transforms.Resample(sample_rate, self.sr)(waveform[0,:].view(1,-1))
        transformed = transformed[:,:self.sr*self.input_length]

        return img, transformed

    def __len__(self):
        return len(self.data)

    def sample_by_MSD_id(self, MSD_id):
        track = self.data.loc[self.data['MSD_track_id'] == MSD_id]
        img_path = track.values[0][1]
        img = self.trans(Image.open(img_path).convert("RGB"))

        audio_path = os.path.join(self.MSD_path, self.id_to_path[self.MSD_id_to_7D_id[MSD_id]])
        waveform, sample_rate = torchaudio.load(audio_path) 
        transformed = torchaudio.transforms.Resample(sample_rate, 16000)(waveform[0,:].view(1,-1))

        return img, transformed
