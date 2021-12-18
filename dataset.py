from numpy.lib.type_check import _imag_dispatcher
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import PIL.Image as Image
from sklearn.preprocessing import MultiLabelBinarizer

import numpy as np

class ImageGenreDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
        # self.mlb = MultiLabelBinarizer()
        self.trans = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        # self.mlb.fit([label_list])
    def __getitem__(self, index):
        album = self.data.iloc[index]
        img_path = album['img_path']
        img = self.trans(Image.open(img_path).convert("RGB"))
        return img
    def __len__(self):
        return len(self.data)

class AlbumAudioDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        track = self.data.iloc[index]
        img_path = track['album_img_path']
        audio_path = track['MSD_track_id']
        img = np.load(img_path)
        audio = np.load(audio_path)
        return img, audio

    def __len__(self):
        return len(self.data)

class TripletDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        track = self.data.iloc[index]
        audio_path = track['MSD_track_id']
        audio = np.load(audio_path)

        pos_path = track['album_img_path']
        pos_z = np.load(pos_path)

        neg_sample = self.data.loc[self.data['amazon_id'] != track['amazon_id']].sample(1)
        neg_path = neg_sample.values[0][1]
        neg_z = np.load(neg_path)

        return audio, pos_z, neg_z

    def __len__(self):
        return len(self.data)