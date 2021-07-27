from re import L
import tensorflow as tf
from tensorflow import keras
from tensorflow import data
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.numpy_ops.np_math_ops import isnan
import config
import time
from config import args_setting
from dataset_tensorflow import RoadSequenceDataset, RoadSequenceDatasetList
from model_tensorflow import generate_model
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import tensorflow as tf
# from torchvision import transforms
# from torch.optim import lr_scheduler


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
def train(args, epoch, model, train_loader, optimizer, criterion):
    since = time.time()
    for batch_idx,  sample_batched in enumerate(train_loader):
        data, target = sample_batched['data'], sample_batched['label']
        
        with tf.GradientTape() as tape:
            output, _ = model(data)
            # print(model.summary())
            output = tf.convert_to_tensor(output)
            # output = tf.keras.layers.Softmax(axis=-1)(output)
            # print(target.shape, output.shape)
            
            # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)

            weights = tf.cast(target, dtype=tf.float32) + 0.02
            loss = criterion(target, output, sample_weight=weights)
            # loss = tf.math.multiply(loss, weights)
            
            # print(loss)
            # loss = tf.clip_by_value(loss, clip_value_min=0, clip_value_max=float("inf"))
            loss = tf.reduce_mean(loss)
            # print(loss)
            
        gradients = tape.gradient(loss, model.trainable_weights)
        # print(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), train_loader.dataset_size,
                100. * batch_idx / len(train_loader), loss))


    time_elapsed = time.time() - since
    print('Train Epoch: {} complete in {:.0f}m {:.0f}s'.format(epoch,
        time_elapsed // 60, time_elapsed % 60))

def val(args, model, val_loader, criterion, epoch):

    loss1= 0
    correct = 0

    for batch_idx, sample_batched in enumerate(val_loader):
        data, target = sample_batched['data'], sample_batched['label']
        
        
        output, _ = model(data)
        # print(model.summary())
        output = tf.convert_to_tensor(output)
        # output = tf.keras.layers.Softmax(axis=-1)(output)
        # print(target.shape, output.shape)
        
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=output)
        # loss = criterion(target, output)
        weights = tf.cast(target, dtype=tf.float32) + 0.02
        loss = tf.math.multiply(loss, weights)
        
        # print(loss)
        # loss = tf.clip_by_value(loss, clip_value_min=0, clip_value_max=float("inf"))
        loss1 += tf.reduce_mean(loss)

        # pred = tf.math.reduce_max(output, axis = 3, keepdims=True)
        pred = tf.math.argmax(output, axis=3)
        temp = tf.cast(tf.math.equal(pred, tf.cast(target , dtype=tf.int64)), dtype = tf.int32)
        correct += tf.math.reduce_sum(temp)
        
        if batch_idx % (args.log_interval * 10) == 0:
            print(f"Batch[{batch_idx}] Accuracy: {100. * int(correct) / (val_loader.dataset_size * config.label_height * config.label_width):.5f}")
            
    loss1 /= (val_loader.dataset_size/args.test_batch_size)
    val_acc = 100. * int(correct) / (val_loader.dataset_size * config.label_height * config.label_width)
    print('\nAverage loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        loss1, int(correct), (val_loader.dataset_size * config.label_height * config.label_width), val_acc))
    # model.save(model, '%s.pth'%val_acc)
    model.save_weights('epoch-{}_acc-{}'.format(epoch, val_acc))


if __name__ == '__main__':
    args = args_setting()
    tf.random.set_seed(args.seed)
    # use_cuda = args.cuda and tf.test.is_gpu_available()
    # device = tf.device("gpu" if use_cuda else "cpu")

    # turn image into floatTensor
    # op_tranforms = image.img_to_array()

    # load data for batches, num_workers for multiprocess
    if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
        train_loader = RoadSequenceDatasetList(file_path=config.train_path, batch_size= args.batch_size, shuffle= True)
        val_loader = RoadSequenceDatasetList(file_path=config.val_path, batch_size= args.test_batch_size, shuffle= True)
    else:
        train_loader = RoadSequenceDataset(file_path=config.train_path, batch_size= args.batch_size, shuffle= True)
        val_loader = RoadSequenceDataset(file_path=config.val_path, batch_size= args.test_batch_size, shuffle= True)

    #load model
    model = generate_model(args)
    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(args.lr, decay_steps=1, decay_rate=0.5, staircase=True)    
    optimizer = tf.keras.optimizers.Adam(learning_rate=scheduler)


    
    # class_weight = tf.convert_to_tensor(config.class_weight)
    class_weight = tf.convert_to_tensor([3.,3.,3.])

    # criterion = tf.nn.weighted_cross_entropy_with_logits(pos_weight=class_weight, name=None)
    # criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy()
    # criterion = tf.keras.losses.CategoricalCrossentropy()
    # criterion = tf.nn.sparse_softmax_cross_entropy_with_logits()
    # criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    best_acc = 0

    # pretrained_dict = torch.load(config.pretrained_path)
    # model_dict = model.state_dict()

    # pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    # model_dict.update(pretrained_dict_1)
    # model.load_state_dict(model_dict)

    # train
    
    
    # print(train_loader[0]['data'].shape)


    for epoch in range(1, args.epochs+1):
        train(args, epoch, model, train_loader, optimizer, criterion)
        val(args, model, val_loader, criterion, epoch)
    model.save_weights('model_epoch_5')