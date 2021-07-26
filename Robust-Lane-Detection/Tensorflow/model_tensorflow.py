import tensorflow as tf
import config
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from utils_tensorflow import *
import operator
# from config import args_setting
import numpy as np

def unpool(value, name='unpool'):
    """N-dimensional version of the unpooling operation from
    https://www.robots.ox.ac.uk/~vgg/rg/papers/Dosovitskiy_Learning_to_Generate_2015_CVPR_paper.pdf

    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    # print(tf.name_scope)
    with tf.name_scope(name) as scope:
        sh = value.get_shape().as_list()
        print(sh)
        dim = len(sh[1:-1])
        print(dim)
        out = (tf.reshape(value, [-1] + sh[-dim:]))
        # print(out)
        for i in range(dim, 0, -1):
            # print(i)
            out = tf.concat([out, tf.zeros_like(out)], i)
        print(out)
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        print(out_size)
        out = tf.reshape(out, out_size, name=scope)
    return out



class UNet_ConvLSTM(Model):
    def __init__(self, n_channels, n_classes):
        super(UNet_ConvLSTM, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        self.convlstm = ConvLSTM(input_size=(8,16),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

    def call(self, x):
        x = tf.unstack(x, axis=1)
        data = []
        for item in x:
            x1 = self.inc(item)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            data.append(tf.expand_dims(x5, axis = 0))
        data = layers.concatenate(data, axis=0)
        lstm, _ = self.convlstm(data)
        test = lstm[-1][ -1,:, :, :, :]
        x = self.up1(test, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x, test



class UNet(Model):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def call(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class SegNet_ConvLSTM(Model):
    def __init__(self):
        super(SegNet_ConvLSTM,self).__init__()
        self.vgg16_bn = tf.keras.applications.VGG16(include_top=False)
        self.relu = layers.ReLU(),
        self.index_MaxPool = layers.MaxPooling2D(pool_size=(2,2), strides=2, data_format = 'channels_last')
        self.index_UnPool = unpool(kernel_size=2, stride=2)
        # net struct
        self.conv1_block = keras.Sequential([self.vgg16_bn.layers[0],  # input layer
                                        self.vgg16_bn.layers[1],  # conv2d(3,64,(3,3))
                                        self.vgg16_bn.layers[2]  # conv2d(3,64,(3,3))
                                        ])
        self.conv2_block = keras.Sequential([self.vgg16_bn.layers[4],
                                        self.vgg16_bn.layers[5]
                                        ])
        self.conv3_block = keras.Sequential([self.vgg16_bn.layers[7],
                                        self.vgg16_bn.layers[8],
                                        self.vgg16_bn.layers[9]
                                        ])
        self.conv4_block = keras.Sequential([self.vgg16_bn.layers[11],
                                        self.vgg16_bn.layers[12],
                                        self.vgg16_bn.layers[13]
                                        ])
        self.conv5_block = keras.Sequential([self.vgg16_bn.layers[15],
                                        self.vgg16_bn.layers[16],
                                        self.vgg16_bn.layers[17]
                                        ])

        self.upconv5_block = keras.Sequential(
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu
                                        )
        self.upconv4_block = keras.Sequential(
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(256, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu
                                        )
        self.upconv3_block = keras.Sequential(
                                        layers.Conv2D(256, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(256, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(128, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu
                                        )
        self.upconv2_block = keras.Sequential(
                                        layers.Conv2D(128, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(64, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu
                                        )
        self.upconv1_block = keras.Sequential(
                                        layers.Conv2D(64, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(config.class_num, (3, 3), padding='same'),
                                        )
        self.convlstm = ConvLSTM(input_size=(4,8),
                                 input_dim=512,
                                 hidden_dim=[512, 512],
                                 kernel_size=(3,3),
                                 num_layers=2,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)
    def call(self, x):
        x = tf.unstack(x, axis=1)
        data = []
        for item in x:
            f1, idx1 = self.index_MaxPool(self.conv1_block(item))
            f2, idx2 = self.index_MaxPool(self.conv2_block(f1))
            f3, idx3 = self.index_MaxPool(self.conv3_block(f2))
            f4, idx4 = self.index_MaxPool(self.conv4_block(f3))
            f5, idx5 = self.index_MaxPool(self.conv5_block(f4))
            data.append(tf.expand_dims(f5, axis = 0))
        data = layers.concatenate(data, axis=0)
        lstm, _ = self.convlstm(data)
        test = lstm[0][-1,:,:,:,:]
        up6 = self.index_UnPool(test,idx5)
        up5 = self.index_UnPool(self.upconv5_block(up6), idx4)
        up4 = self.index_UnPool(self.upconv4_block(up5), idx3)
        up3 = self.index_UnPool(self.upconv3_block(up4), idx2)
        up2 = self.index_UnPool(self.upconv2_block(up3), idx1)
        up1 = self.upconv1_block(up2)
        return tf.nn.log_softmax(up1, axis=1)


class SegNet(Model):
    def __init__(self):
        super(SegNet,self).__init__()
        self.vgg16_bn = tf.keras.applications.VGG16(include_top=False)
        self.relu = layers.ReLU(),
        self.index_MaxPool = layers.MaxPooling2D(pool_size=(2,2), strides=2, data_format = 'channels_last')
        self.index_UnPool = unpool(kernel_size=2, stride=2)
        # net struct

        self.conv1_block = keras.Sequential([self.vgg16_bn.layers[0],  # input layer
                                        self.vgg16_bn.layers[1],  # conv2d(3,64,(3,3))
                                        self.vgg16_bn.layers[2]  # conv2d(3,64,(3,3))
                                        ])
        self.conv2_block = keras.Sequential([self.vgg16_bn.layers[4],
                                        self.vgg16_bn.layers[5]
                                        ])
        self.conv3_block = keras.Sequential([self.vgg16_bn.layers[7],
                                        self.vgg16_bn.layers[8],
                                        self.vgg16_bn.layers[9]
                                        ])
        self.conv4_block = keras.Sequential([self.vgg16_bn.layers[11],
                                        self.vgg16_bn.layers[12],
                                        self.vgg16_bn.layers[13]
                                        ])
        self.conv5_block = keras.Sequential([self.vgg16_bn.layers[15],
                                        self.vgg16_bn.layers[16],
                                        self.vgg16_bn.layers[17]
                                        ])

        self.upconv5_block = keras.Sequential(
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu
                                        )
        self.upconv4_block = keras.Sequential(
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(512, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(256, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu
                                        )
        self.upconv3_block = keras.Sequential(
                                        layers.Conv2D(256, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(256, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(128, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu
                                        )
        self.upconv2_block = keras.Sequential(
                                        layers.Conv2D(128, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(64, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu
                                        )
        self.upconv1_block = keras.Sequential(
                                        layers.Conv2D(64, (3, 3), padding='same'),
                                        layers.BatchNormalization(epsilon=1e-05, momentum=0.9),
                                        self.relu,
                                        layers.Conv2D(config.class_num, (3, 3), padding='same'),
                                        )
    def call(self, x):
        f1, idx1 = self.index_MaxPool(self.conv1_block(x))
        f2, idx2 = self.index_MaxPool(self.conv2_block(f1))
        f3, idx3 = self.index_MaxPool(self.conv3_block(f2))
        f4, idx4 = self.index_MaxPool(self.conv4_block(f3))
        f5, idx5 = self.index_MaxPool(self.conv5_block(f4))
        up6 = self.index_UnPool(f5,idx5)
        up5 = self.index_UnPool(self.upconv5_block(up6), idx4)
        up4 = self.index_UnPool(self.upconv4_block(up5), idx3)
        up3 = self.index_UnPool(self.upconv3_block(up4), idx2)
        up2 = self.index_UnPool(self.upconv2_block(up3), idx1)
        up1 = self.upconv1_block(up2)

        return tf.nn.log_softmax(up1, axis=1)



def generate_model(args):

    use_cuda = args.cuda and tf.test.is_gpu_available()
    device = tf.device("gpu" if use_cuda else "cpu")

    assert args.model in [ 'UNet-ConvLSTM', 'SegNet-ConvLSTM', 'UNet', 'SegNet']
    if args.model == 'SegNet-ConvLSTM':
        model = SegNet_ConvLSTM()
    elif args.model == 'SegNet':
        model = SegNet()
    elif args.model == 'UNet-ConvLSTM':
        model = UNet_ConvLSTM(config.img_channel, config.class_num)
    elif args.model == 'UNet':
        model = UNet(config.img_channel, config.class_num)
    return model
