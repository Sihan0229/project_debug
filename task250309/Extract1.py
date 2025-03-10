from keras import backend as K
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import preprocess_input
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import glob
from tqdm import tqdm
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import shutil
import time
# from cv2 import cv2
import cv2
import h5py
os.environ['KERAS_BACKEND']='tensorflow'
#######################################
# 在训练的时候置为1
K.set_learning_phase(1)
#######################################

EPOCHS =30
RANDOM_STATE = 2020
learning_rate = 0.003
# 分类类别
# CLASS_NUM = 21
CLASS_NUM = 200
# os.chdir(r'E:\receipt_angle')

TRAIN_DIR = "./Caltech200/train/"
VALID_DIR = "./Caltech200/valid/"


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
                              verbose=1, mode='auto')
    return [lr_reduce, msave, earlystop]

def add_new_last_layer(base_model, nb_classes, drop_rate=0.):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    features = Dense(512)(x)
    predictions = Dense(nb_classes, activation='sigmoid')(features)  # new softmax layer
    model = Model(input=base_model.input, output=predictions)
    return model

def get_model(IN_WIDTH, IN_HEIGHT):
    """
    获得模型
    """
    base_model = DenseNet169(include_top=False,
                             weights='imagenet',
                             input_shape=(IN_WIDTH, IN_HEIGHT, 3))
    # model=base_model.add(Flatten())
    model = add_new_last_layer(base_model, CLASS_NUM)
    #model=base_model
    model.compile(optimizer=SGD(lr=learning_rate, momentum=0.9),
                  loss="binary_crossentropy",
                  metrics=['accuracy'])
    # loss=[mycrossentropy()], metrics=['accuracy']) #设置损失是否为交叉熵
    model.summary() #输出参数Param计算过程
    return model

def train_model(save_model_path, BATCH_SIZE, IN_SIZE):
    IN_WIDTH, IN_HEIGHT = IN_SIZE
    callbacks = get_callbacks(filepath=save_model_path, patience=3)
    model = get_model(IN_WIDTH, IN_HEIGHT)
    # 角度训练模型修改+
    train_datagen = ImageDataGenerator(
        # （1）图片生成器，负责生成一个批次一个批次的图片，以生成器的形式给模型训练；
        # （2）对每一个批次的训练图片，适时地进行数据增强处理（data augmentation）；
        preprocessing_function=preprocess_input,
        horizontal_flip=False,  # 如果角度不是一个训练特征，这个设置为true
        vertical_flip=False,
        shear_range=0.1
    )

    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    generator = ImageDataGenerator(

    )

    train_generator = train_datagen.flow_from_directory(
        directory=TRAIN_DIR,
        target_size=(IN_WIDTH, IN_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_STATE,
        interpolation='bilinear',  # PIL默认插值下采样的时候会模糊
    )

    valid_generator = valid_datagen.flow_from_directory(
        directory=VALID_DIR,
        target_size=(IN_WIDTH, IN_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        seed=RANDOM_STATE,
        interpolation='bilinear',  # PIL默认插值下采样的时候会模糊
    )
    print(valid_generator.samples) #300
    print(valid_generator.samples // BATCH_SIZE)    #9

    model.fit_generator(
        train_generator,
        steps_per_epoch=1 * (train_generator.samples // BATCH_SIZE + 1),
        epochs=EPOCHS,
        max_queue_size=1000,
        workers=2,
        verbose=1,
        validation_data=valid_generator,  # valid_generator,
        validation_steps=valid_generator.samples // BATCH_SIZE,
        # valid_generator.samples // BATCH_SIZE + 1, #len(valid_datagen)+1,
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
    ''''
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
    try:
        x = load_img(path=image_path, target_size=(
            IN_HEIGHT, IN_WIDTH, 3), interpolation='bilinear')
    except Exception as e:
        print('\n' + image_path + ": 文件出错 \n")
        return None
    x = img_to_array(x)
    x = preprocess_input(x)
    x = x[None]

    featextract = K.function([model.get_input_at(0)],[model.layers[-2].output])
    feats = featextract([x])[0]
    return feats

def extract(model,img_dir, IN_SIZE,target_path):
    image_paths = []
    #我的10类图像
    for filename in range(CLASS_NUM):
        picnames = os.listdir(os.path.join(img_dir,str(filename)))
        for picname in picnames:
            image_paths.append(os.path.join(img_dir,str(filename),picname))
    paths = []
    feats = []
    for image_path in tqdm(image_paths):
        result_int = predict(model,image_path, IN_SIZE)
        feats.append(result_int)
        path, _ = os.path.split(image_path)
        _, gt = os.path.split(path)
        paths.append(image_path)

    output = target_path
    h5y = h5py.File(output, 'w')
    h5y.create_dataset('paths',data=paths)
    h5y.create_dataset('feats',data=feats)
    h5y.close()
    #预测所属类别和实际类别

def main():
    # ～～～～～～～～～～～～～～～～～  模型训练  ～～～～～～～～～～～～～～～～～
    IN_SIZE = (256, 256)    #图像大小·
    weights_path = r'./200/Model/512Dim.hdf5'     #训练后的模型保存位置
    # train_model(save_model_path=weights_path, BATCH_SIZE=32, IN_SIZE=IN_SIZE)
# ～～～～～～～～～～～～～～～～～  加载模型提取特征  ～～～～～～～～～～～～～～～～～
    model = load_weight(weights_path, IN_SIZE)
    img_dir = r'./Caltech200/train/'  #图像dir
    target = './200/512features.h5'  #图像特征保存路径
    extract(model,img_dir, IN_SIZE, target)

if __name__ == '__main__':
    main()