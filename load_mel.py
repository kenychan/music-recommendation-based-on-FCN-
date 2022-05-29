import os
import pandas as pd
import torch
import numpy as np


class MelDataset():
    """Face Landmarks dataset."""

    def __init__(self, annotation, audio_info, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audio_annotaion = pd.read_csv(annotation,low_memory=False)
        self.audio_info = pd.read_csv(audio_info,low_memory=False)
        self.transform = transform

    def __len__(self):
        return len(self.audio_annotaion)  # ok

    def __getitem__(self, idx):  # to support the indexing such that dataset[i] can be used to get iith sample.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        try:
            data = self.audio_info.iloc[idx, 11]  # 11 = mel_data path
            mel_data = np.loadtxt(data, delimiter=",",dtype=np.float16).reshape(1,96,1366)#for batching it needs to be 3d

            mel_labels = self.audio_annotaion.iloc[idx, 1:]
            mel_labels = np.array(mel_labels, dtype=np.float16)

        # mel_labels = mel_labels .astype('float').reshape(-1, 2)#only define the column size, row can be random
        # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape

            sample = {'mel': mel_data, 'label': mel_labels}

            if self.transform:
                sample = self.transform(sample)

            return sample
        except ValueError:
            print("erro clip id: " ,self.audio_annotaion.iloc[idx, 0])



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        mel, label = torch.from_numpy(sample['mel']), torch.from_numpy(sample['label'])

        return {'mel': mel,
                'label': label}

# dataset = MelDataset(annotation="data/genre_ins_key.csv",audio_info="data/clip_info_with mel_data_path.csv",
#                                    transform=ToTensor())
