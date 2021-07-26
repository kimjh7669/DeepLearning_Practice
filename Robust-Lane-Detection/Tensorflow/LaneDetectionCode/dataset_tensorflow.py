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

    def __init__(self, file_path, batch_size, shuffle = True):
        self.batch_size = batch_size
        self.img_list = readTxt(file_path)
        self.dataset_size = len(self.img_list)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.dataset_size) / self.batch_size)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.dataset_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        list_id = [self.img_list[k] for k in indexes]
        batch_data = []
        batch_label = []
        if_first = True
        for img_path_list in list_id:
            # img_path_list = self.img_list[idx]
            data = []
            # print('hello')
            for i in range(5):
                # print(img_path_list[i])
                temp = np.array(image.img_to_array(Image.open(img_path_list[i]))) / 255.
                data.append(tf.expand_dims(temp, axis=0))
                
            data = tf.keras.layers.concatenate(data, 0)
            # print(data.shape)
            label = np.array(image.img_to_array(Image.open(img_path_list[5]))) / 255.
            
            label = tf.squeeze(label)
            label = tf.cast(label, dtype = tf.int32)
            
            if if_first == True:
                temp = np.array(data)
                batch_data = (tf.expand_dims(temp, axis = 0))
                temp = np.array(label)
                batch_label = tf.expand_dims(temp, axis = 0)
            else:
                temp = tf.expand_dims(np.array(data), axis = 0)
                batch_data = tf.keras.layers.concatenate([batch_data, temp], axis = 0)
                temp = tf.expand_dims(np.array(label), axis = 0)
                batch_label = tf.keras.layers.concatenate([batch_label, temp], axis = 0)
            if_first = False

        sample = {'data': batch_data, 'label': batch_label}

        return sample