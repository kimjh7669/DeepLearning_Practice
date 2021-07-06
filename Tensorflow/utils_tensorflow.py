import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

class double_conv(tf.Module): 
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = keras.Sequential([
            layers.Conv2D(out_ch, (3, 3), padding = "SAME", activation=None), 
            layers.BatchNormalization(momentum=0.9),
            layers.Activation('relu'),
            layers.Conv2D(out_ch, (3, 3), padding = "SAME", activation=None), 
            layers.BatchNormalization(momentum=0.9),
            layers.Activation('relu')            
        ])
    def __call__(self, x):
        x = self.conv(x)
        return x

class inconv(tf.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
    def __call__(self, x):
        x = self.conv(x)
        return x

class down(tf.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = keras.Sequential([
            layers.MaxPooling2D(2),
            layers.Conv2D(out_ch, (3, 3), padding = "SAME", activation=None), 
            layers.BatchNormalization(momentum=0.9),
            layers.Activation('relu'),
            layers.Conv2D(out_ch, (3, 3), padding = "SAME", activation=None), 
            layers.BatchNormalization(momentum=0.9),
            layers.Activation('relu') 
        ])

    def __call__(self, x):
        x = self.mpconv(x)
        return x
        

class up(tf.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = layers.UpSampling2D(size = (2,2), interpolation='bilinear')
        else:
            self.up = layers.Conv2DTranspose(in_ch//2, 2, stride=(2, 2))

        self.conv = double_conv(in_ch, out_ch)

    def __call__(self, x1, x2):
        x1 = self.up(x1) 
        diffX = x1.shape[1] - x2.shape[1] # height ?
        diffY = x1.shape[2] - x2.shape[2] # width ?
        padding = [[0,0], [(diffX)// 2, int((diffX + 1)/ 2)], [diffY // 2, int((diffY+1) / 2)], [0,0]]
        x2 = tf.pad(x2, padding)
        x = layers.concatenate([x2, x1], axis=3)
        x = self.conv(x)
        return x

class outconv(tf.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = layers.Conv2D(out_ch, (1, 1), padding = "SAME", activation=None)
    def __call__(self, x):
        x = self.conv(x)
        return x


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
                              padding="valid",
                              use_bias=self.bias)
        ])

    def __call__(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = layers.concatenate([input_tensor, h_cur], axis=3)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        for_split_list = []
        
        for i in range(combined_conv.shape[3] // self.hidden_dim):
            for_split_list.append(self.hidden_dim)

        cc_i, cc_f, cc_o, cc_g = tf.split(combined_conv, for_split_list, axis=3) # split 
        i = tf.sigmoid(cc_i)
        f = tf.sigmoid(cc_f)
        o = tf.sigmoid(cc_o)
        g = tf.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * tf.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size):
        return (tf.zeros([batch_size, self.height, self.width, self.hidden_dim]),
                tf.zeros([batch_size, self.height, self.width, self.hidden_dim]))


class ConvLSTM(tf.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        # self.cell_list = nn.ModuleList(cell_list)
        self.cell_list = cell_list
        
    def __call__(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w) -> t = seq num
            # (t, b, h, w, c) -> (b, t, h, w, c)
            input_tensor = tf.transpose(input_tensor, [1, 0, 2, 3, 4])

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.shape[0])

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])


                output_inner.append(h)

            layer_output = tf.stack(output_inner, axis=1)
            cur_layer_input = layer_output
            
            layer_output = tf.transpose(layer_output, [1, 0, 2, 3, 4])
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
