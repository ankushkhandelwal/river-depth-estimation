# merge cloud and size info into a single panda frame
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.initializers

def img_img(pretrained_weights = None,input_size = (96,96,9)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_1')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='layer_3')(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_4')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_5')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='layer_6')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_7')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_8')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='layer_9')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_10')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_11')(conv4)
    drop4 = Dropout(0.5,name='layer_12')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='layer_13')(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_14')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_15')(conv5)
    drop5 = Dropout(0.5,name='layer_16')(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_1')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3,name='Dlayer_2')
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_3')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_4')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_5')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3,name='Dlayer_6')
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_7')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_8')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_9')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3,name='Dlayer_10')
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_11')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_12')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_13')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3,name='Dlayer_14')
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_15')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_16')(conv9)
    conv9 = Conv2D(9, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Dlayer_17')(conv9)
    conv10 = Conv2D(9, 1, activation = 'sigmoid',name='Dlayer_18')(conv9)

    model = Model(input = inputs, output = conv10)

    if(pretrained_weights):
        model.load_weights(pretrained_weights, by_name=True)
    # model  = multi_gpu_model(model,gpus=1)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['mean_squared_error'])

    return model


def img_label(pretrained_weights = None,input_size = (96,96,9),type='random'):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_1')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='layer_3')(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_4')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_5')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='layer_6')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_7')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_8')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='layer_9')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_10')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_11')(conv4)
    drop4 = Dropout(0.5,name='layer_12')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='layer_13')(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_14')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_15')(conv5)
    drop5 = Dropout(0.5,name='layer_16')(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_1')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3,name='Llayer_2')
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_3')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_4')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_5')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3,name='Llayer_6')
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_7')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_8')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_9')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3,name='Llayer_10')
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_11')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_12')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_13')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3,name='Llayer_14')
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_15')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_16')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_17')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid',name='Llayer_18')(conv9)

    model = Model(input = inputs, output = conv10)

    if type=='init':
        if(pretrained_weights):
            model.load_weights(pretrained_weights, by_name=True)
    if type=='frozen':
        if(pretrained_weights):
        	model.load_weights(pretrained_weights, by_name=True)
        for layer in model.layers[1:17]:
            layer.trainable = False
    if type=='random':
        if(pretrained_weights):
            model.load_weights(pretrained_weights, by_name=True)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


def gray_label(pretrained_weights = None,input_size = (96,96,1),type='random'):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_1')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='layer_3')(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_4')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_5')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='layer_6')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_7')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_8')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='layer_9')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_10')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_11')(conv4)
    drop4 = Dropout(0.5,name='layer_12')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='layer_13')(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_14')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_15')(conv5)
    drop5 = Dropout(0.5,name='layer_16')(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_1')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3,name='Llayer_2')
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_3')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_4')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_5')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3,name='Llayer_6')
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_7')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_8')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_9')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3,name='Llayer_10')
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_11')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_12')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_13')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3,name='Llayer_14')
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_15')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_16')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_17')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid',name='Llayer_18')(conv9)

    model = Model(input = inputs, output = conv10)

    if type=='init':
        if(pretrained_weights):
            model.load_weights(pretrained_weights, by_name=True)
    if type=='frozen':
        if(pretrained_weights):
        	model.load_weights(pretrained_weights, by_name=True)
        for layer in model.layers[1:17]:
            layer.trainable = False
    if type=='random':
        if(pretrained_weights):
            model.load_weights(pretrained_weights, by_name=True)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model




def gray_gray(pretrained_weights = None,input_size = (96,96,1),type='random'):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_1')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_2')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='layer_3')(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_4')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_5')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='layer_6')(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_7')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_8')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='layer_9')(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_10')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_11')(conv4)
    drop4 = Dropout(0.5,name='layer_12')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='layer_13')(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_14')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_15')(conv5)
    drop5 = Dropout(0.5,name='layer_16')(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_1')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3,name='Llayer_2')
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_3')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_4')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_5')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3,name='Llayer_6')
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_7')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_8')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_9')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3,name='Llayer_10')
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_11')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_12')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_13')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3,name='Llayer_14')
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_15')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_16')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_17')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid',name='Llayer_18')(conv9)

    model = Model(input = inputs, output = conv10)



    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['mean_squared_error'])
    return model



def img_label_bayesian(pretrained_weights = None,input_size = (96,96,9),type='random'):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_1')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_2')(conv1)
    drop1 = Dropout(0.5,name='layer_3')(conv1,training=True)
    pool1 = MaxPooling2D(pool_size=(2, 2),name='layer_4')(drop1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_5')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_6')(conv2)
    drop2 = Dropout(0.5,name='layer_7')(conv2,training=True)
    pool2 = MaxPooling2D(pool_size=(2, 2),name='layer_8')(drop2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_9')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_10')(conv3)
    drop3 = Dropout(0.5,name='layer_11')(conv3,training=True)
    pool3 = MaxPooling2D(pool_size=(2, 2),name='layer_12')(drop3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_13')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_14')(conv4)
    drop4 = Dropout(0.5,name='layer_15')(conv4,training=True)
    pool4 = MaxPooling2D(pool_size=(2, 2),name='layer_16')(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_17')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='layer_18')(conv5)
    drop5 = Dropout(0.5,name='layer_19')(conv5,training=True)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_1')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3,name='Llayer_2')
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_3')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_4')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_5')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3,name='Llayer_6')
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_7')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_8')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_9')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3,name='Llayer_10')
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_11')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_12')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_13')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3,name='Llayer_14')
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_15')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_16')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name='Llayer_17')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid',name='Llayer_18')(conv9)

    model = Model(input = inputs, output = conv10)

    if type=='init':
        if(pretrained_weights):
            model.load_weights(pretrained_weights, by_name=True)
    if type=='frozen':
        if(pretrained_weights):
        	model.load_weights(pretrained_weights, by_name=True)
        for layer in model.layers[1:17]:
            layer.trainable = False
    if type=='random':
        if(pretrained_weights):
            model.load_weights(pretrained_weights, by_name=True)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
