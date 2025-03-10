from keras import backend as K
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.layers import Dropout,Convolution2D, MaxPooling2D, BatchNormalization,Activation,Flatten
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import glob
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import shutil
import time
import cv2
from data_generator import DataGenerator
import h5py
from image_processing import  image_process
# 图像分类和哈希特征学习的深度学习模型，核心目标是将图像映射为低维哈希编码（24维）
#######################################
# 在训练的时候置为1
K.set_learning_phase(1)
#######################################

EPOCHS = 50
RANDOM_STATE = 2021
learning_rate = 0.003
# 分类类别

Dimension_count=24

TRAIN_TXT = r"train.txt"
VALID_TXT = r"valid.txt"
labelHsh_path = './Data/labelHsh24_8_254.npy'


def get_callbacks(filepath, patience=3):
    '''
    学习率调整
    monitor：监测的值，可以是accuracy，val_loss,val_accuracy
    factor：缩放学习率的值，学习率将以lr = lr*factor的形式被减少
    patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    mode：‘auto’，‘min’，‘max’之一 默认‘auto’就行
    epsilon：阈值，用来确定是否进入检测值的“平原区”
    cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
    min_lr：学习率最小值，能缩小到的下限'''
    lr_reduce = ReduceLROnPlateau(monitor='val_accuracy',
                                  factor=0.1,
                                  epsilon=1e-5,
                                  patience=patience,
                                  verbose=1,
                                  min_lr=0.00001)
    # 该回调函数将在每个epoch后保存模型到filepath
    msave = ModelCheckpoint(
        filepath, monitor='val_accuracy', save_best_only=True, verbose=1)
    '''
    filename：字符串，保存模型的路径    monitor：需要监视的值 verbose：信息展示模式，0或1
    save_best_only：当设置为True时，将只保存在验证集上性能最好的模型

    mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，
    例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
    save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
    period：CheckPoint之间的间隔的epoch数
    '''
    earlystop = EarlyStopping(monitor='val_accuracy',
                              min_delta=0,
                              patience=patience * 3 + 2,
                              verbose=1,
                              mode='auto')
    return [lr_reduce, msave, earlystop]




def add_new_last_layer(base_model, dimension, drop_rate=0.):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    # x = base_model.output
    # x = Dropout(0.5)(x)
    # x = Convolution2D(48,1,1)(x)
    # x = BatchNormalization(axis=1, epsilon=1.001e-5)(x)
    # x = Activation('relu')(x)
    # Hashx = GlobalAveragePooling2D()(x)

    x = base_model.output

    # # 2
    # x = Convolution2D(1024, (3, 3), padding='same')(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Convolution2D(512, (1, 1), padding='same')(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dropout(0.5)(x)
    # predictions = Dense(nb_classes, activation='sigmoid')(x)

    # # 3
    # x = Convolution2D(1024, (1, 1))(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Flatten()(x)
    # x = Dense(512)(x)
    # x = Activation('relu')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    # predictions = Dense(nb_classes, activation='sigmoid')(x)



    #4
    x = Dropout(0.5)(x)
    x = Convolution2D(64, (1, 1))(x)
    x = BatchNormalization(axis=1, epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    Hashx = GlobalAveragePooling2D()(x)
    x = Dense(512)(Hashx)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)
    predictions = Dense(dimension, activation='sigmoid')(x)

    model = Model(input=base_model.input, output=predictions)
    return model


def get_model(IN_WIDTH, IN_HEIGHT):
    """
    获得模型
    """
    base_model = DenseNet169(
        include_top=False, weights='imagenet', input_shape=(IN_WIDTH, IN_HEIGHT, 3))

    model = add_new_last_layer(base_model, Dimension_count)
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9),
                  loss="binary_crossentropy", metrics=['accuracy'])
    # loss=[mycrossentropy()], metrics=['accuracy']) #设置损失是否为交叉熵
    model.summary()

    return model


def train_model(save_model_path, BATCH_SIZE, IN_SIZE):
    IN_WIDTH, IN_HEIGHT = IN_SIZE

    callbacks = get_callbacks(filepath=save_model_path, patience=3)
    model = get_model(IN_WIDTH, IN_HEIGHT)

    train_gen = DataGenerator(dataset_path=TRAIN_TXT,
                              labelHsh_path=labelHsh_path,
                              batch_size=BATCH_SIZE,
                              image_size=IN_SIZE,
                              horizontal_flip=False,
                              vertical_flip=False,
                              rotation_range=10,
                              shear_range=0.1,
                              )
    valid_gen = DataGenerator(dataset_path=VALID_TXT,
                              labelHsh_path=labelHsh_path,
                              batch_size=BATCH_SIZE,
                              image_size=IN_SIZE,
                              )



    model.fit_generator(
        train_gen.get_mini_batch(),
        steps_per_epoch=1 * (train_gen.get_data_number() // BATCH_SIZE + 1),
        epochs=EPOCHS,
        max_queue_size=1000,
        workers=2,
        verbose=1,
        validation_data=valid_gen.get_mini_batch(), # valid_generator,
        validation_steps=valid_gen.get_data_number() // BATCH_SIZE,
        callbacks=callbacks
    )



def load_weight(weight_path, IN_SIZE):
    '''
    Function :
        加载权重防止多次加载
    Args :
        weight_path : 权重路径
        IN_SIZE : 训练时输入尺度大小
    Return :
        权重模型
    Raises :
        无
    '''
    IN_WIDTH, IN_HEIGHT = IN_SIZE
    K.set_learning_phase(0)
    model = get_model(IN_WIDTH, IN_HEIGHT)
    model.load_weights(weight_path)
    return model



def predict(model, image_path, IN_SIZE):
    '''
    Function :
        预测小票类别
    Args :
        model : 加载的权重模型
        image_path :小票路径
    Return :
        分类结果
    Raises :
        文件插值时出错，返回None
    '''
    IN_WIDTH, IN_HEIGHT = IN_SIZE


    img = cv2.imread(image_path)
    img = cv2.resize(img, (IN_WIDTH, IN_HEIGHT), interpolation=cv2.INTER_LINEAR)
    img = np.array(img)

    # 数据增强
    img = image_process(x_img=img, horizontal_flip=False,
                        vertical_flip=False,
                        rotation_range=10,
                        shear_range=0.1)
    x = []
    x.append(img)
    x = np.array(x)

    # 预测

    featextract = K.function([model.get_input_at(0)],[model.layers[-6].output])
    labelextract = K.function([model.layers[-5].input],[model.layers[-1].output])
    feats = featextract([x])[0]
    labels = labelextract([feats])[0]
    Hash = np.concatenate((labels,feats),axis=1)

    return Hash

    # featextract = K.function([model.get_input_at(0)],[model.layers[-1].output])
    # return featextract([x])[0]

def evaluate(model, img_dir, IN_SIZE,target_path):
    image_paths = []
    for filename in range(50):
        picnames = os.listdir(os.path.join(img_dir, str(filename)))
        for picname in picnames:
            image_paths.append(os.path.join(img_dir, str(filename), picname))
    labels = []
    feats = []
    for image_path in tqdm(image_paths):
        result_int = predict(model, image_path, IN_SIZE)
        feats.append(result_int)
        path, _ = os.path.split(image_path)
        _, gt = os.path.split(path)
        labels.append(image_path)

    feats = np.array(feats)
    feats = feats.reshape([feats.shape[0],feats.shape[2]])
    output = target_path
    # h5y = h5py.File(output, 'w')
    # h5y.create_dataset('feats', data=feats)
    # h5y.create_dataset('paths', data=labels)
    # h5y.close()


def main():
    # ～～～～～～～～～～～～～～～～～  trainning  ～～～～～～～～～～～～～～～～～
    IN_SIZE = (256, 256)

    weights_path = r'./50/Model/24_64.hdf5'

    # train_model(save_model_path=weights_path, BATCH_SIZE=32, IN_SIZE=IN_SIZE)

    #-------------------------------extract------------------------------------------
    stime = time.time()
    model = load_weight(weights_path, IN_SIZE)

    RunList=['train']
    for List in RunList:
        img_dir = r'./Caltech50/'+List
        target = './50/Features/'+List+'.h5'
        evaluate(model, img_dir, IN_SIZE, target)

    etime = time.time()
    extractTime = round(etime-stime, 4)
    np.save('./Result/extractTime.npy',extractTime)
    print(extractTime)

if __name__ == '__main__':
    main()
