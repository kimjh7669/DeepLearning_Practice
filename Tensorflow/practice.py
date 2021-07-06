# import tensorflow as tf
# from tensorflow.keras import model

# initial_learning_rate = 0.1
# lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#     initial_learning_rate,
#     decay_steps=100000,
#     decay_rate=0.96,
#     staircase=True)

# model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(data, labels, epochs=5)
from torch.optim import lr_scheduler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.autograd import Variable



onehot_labels = [[0,0,1,0,0],
                  [0,0,0,1,0],
                  [0,1,0,0,0],
                  [1,0,0,0,0]]
labels = np.argmax(onehot_labels, axis=1)
# [2 3 1 0]
logits = [[-1.1258, -1.1524, -0.2506, -0.4339,  0.5988],
          [-1.5551, -0.3414,  1.8530,  0.4681, -0.1577],
          [ 1.4437,  0.2660,  1.3894,  1.5863,  0.9463],
          [-0.8437,  0.9318,  1.2590,  2.0050,  0.0537]]
class_weight = [0.02, 1.02]

# Convert to a tensor that tf can handle
tflabels = tf.constant(labels)
tflabels_oh = tf.constant(onehot_labels)
tflogits = tf.constant(logits)

# tfloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tflabels, logits=tflogits)
# tfloss_oh = tf.nn.softmax_cross_entropy_with_logits(labels=tflabels_oh, logits=tflogits)
tfloss = tf.nn.weighted_cross_entropy_with_logits(pos_weight=class_weight, name=None)
# lossvalue = tfloss.numpy()
# loss_oh_value = tfloss_oh.numpy()

# print('tfloss\t\t', lossvalue)
# print('tfloss_oh\t', loss_oh_value)





# # Convert to a tensor that pytorch can recognize
# ptlabels = torch.tensor(labels).int()
# ptlogits = torch.tensor(logits)


# # ptloss = torch.nn.CrossEntropyLoss(reduce=False)(ptlogits, ptlabels.long())
# # ptloss2 = torch.nn.NLLLoss(reduce=False)(torch.nn.LogSoftmax(dim=-1)(ptlogits), ptlabels.long())
# class_weight_tor = torch.Tensor(class_weight)
# ptloss = torch.nn.CrossEntropyLoss(weight = class_weight_tor, reduce=False)(ptlogits, ptlabels.long())
# # loss = ptloss(ptlogits, ptlabels.long())
# # print('ptloss:\t\t', loss)
# print('ptloss:\t\t', ptloss)
# print('ptloss2:\t', ptloss2)



# print(labels)


