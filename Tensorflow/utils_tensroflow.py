import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class double_conv(keras.Module):
    def __init(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = keras.Sequential(
            layers.Conv2D(out_ch, 3, padding = "SAME", activation=None), # keras에서 input_channel의 수가 중요한가?
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.Conv2D(out_ch, 3, padding = "SAME", activation=None), # keras에서 input_channel의 수가 중요한가?
            layers.BatchNormalization(),
            layers.Activation('relu')            
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(keras.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x

class down(keras.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = keras.Sequential(
            keras.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
        

class up(keras.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = layers.UpSampling2D(size = (2,2), interpolation='bilinear')
        else:
            self.up = layers.Conv2DTranspose(in_ch//2, 2, stride=(2, 2))

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x