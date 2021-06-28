import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from torch.autograd import Variable


class ConvLSTMCell(tf.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias


        self.conv = keras.Sequential([
            layers.ZeroPadding2D(padding=self.padding),
            layers.Conv2D( #in_channels=self.input_dim + self.hidden_dim,
                              filters=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding='valid',
                              use_bias=self.bias)
        ])

    def __call__(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = layers.concatenate([input_tensor, h_cur], axis=3)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        for_split_list = []
        
        for i in range(combined_conv.shape[3] // self.hidden_dim):
            for_split_list.append(self.hidden_dim)
        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, for_split_list, axis=3) # split 확인할 것
        i = tf.sigmoid(cc_i)
        f = tf.sigmoid(cc_f)
        o = tf.sigmoid(cc_o)
        g = tf.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tf.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (tf.zeros(batch_size, self.hidden_dim, self.height, self.width),
                tf.zeros(batch_size, self.hidden_dim, self.height, self.width))

def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

num_layers = 10
input_dim = 5
# hidden_dim
hidden_dim = 5
height, width = 2, 5
kernel_size = (3, 3)
bias = True
kernel_size = _extend_for_multilayer(kernel_size, num_layers)
hidden_dim = _extend_for_multilayer(hidden_dim, num_layers)

cell_list = []

for i in range(0, num_layers):
    # cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]

    cell_list.append(ConvLSTMCell(input_size=(height, width),
                                    input_dim=input_dim,
                                    hidden_dim=hidden_dim[i],
                                    kernel_size=kernel_size[i],
                                    bias=bias))
x = np.arange(1,501,dtype=np.float32).reshape(1,10,2,5,5)
x1 = np.arange(1,51,dtype=np.float32).reshape(1,2,5,5)
x2 = np.arange(1,51,dtype=np.float32).reshape(1,2,5,5)
x = tf.convert_to_tensor(x)
x1 = tf.convert_to_tensor(x1)
x2 = tf.convert_to_tensor(x2)
# print(x1)
for i in range(0, num_layers):
    # # print(x[:, i, :, :, :])
    print(cell_list[i](input_tensor=x[:,i,:,:,:], cur_state = (x1, x2)))

