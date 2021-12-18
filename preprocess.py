import pandas as pd
from preprocessor import Preprocessor

data_path = './MuMu_dataset/rearranged_MuMu_dataset.csv'
data = pd.read_csv(data_path)
preprocessor = Preprocessor(npy_path='./preprocessed_data')
data = preprocessor.preprocessing(data)
data.to_csv(data_path, index = False, header=True)