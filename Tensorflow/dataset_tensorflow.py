import tensorflow as tf
from tensorflow import data
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from PIL import Image
import config
import numpy as np
# from sklearn import preprocessing

def readTxt(file_path):
    img_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list

class RoadSequenceDataset(Sequence):

    def __init__(self, file_path):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = Image.open(img_path_list[4])
        label = Image.open(img_path_list[5])
        data = image.img_to_array(data)
        label = tf.squeeze(image.img_to_array(label))
        sample = {'data': data, 'label': label}
        return sample

class RoadSequenceDatasetList(Sequence):

    def __init__(self, file_path):

        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img_path_list = self.img_list[idx]
        data = []
        print('hello')
        for i in range(5):
            print(img_path_list[i])
            data.append(tf.expand_dims(image.img_to_array(Image.open(img_path_list[i])), axis=0))
        
        data = tf.keras.layers.concatenate(data, 0)
        label = Image.open(img_path_list[5])
        label = tf.squeeze(image.img_to_array(label))
        sample = {'data': data, 'label': label}
        return sample


