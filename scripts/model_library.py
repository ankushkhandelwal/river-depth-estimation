import numpy as np
import os
# import skimage.io as io
# import skimage.transform as trans
# import tensorflow as tf
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import keras.initializers
# import matplotlib.pyplot as plt
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # import md5, sha
# tf.logging.set_verbosity(tf.logging.ERROR)

def cloud_pred(full_band_matrix):

    full_band_matrix=full_band_matrix*1.0/10000
    band_matrix_input=full_band_matrix[:,:,[0,1,3,4,7,8,9,10,11,12]]
    band_matrix_input = np.reshape(band_matrix_input,(1,)+band_matrix_input.shape)
    cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
    cloud_matrix=np.zeros((band_matrix_input.shape[1],band_matrix_input.shape[2],2),float)
    cloud_matrix[:,:,0] = cloud_detector.get_cloud_probability_maps(np.array(band_matrix_input))
    cloud_matrix[:,:,1] = cloud_detector.get_cloud_masks(np.array(band_matrix_input))
    # plt.figure()
    # plt.imshow(cloud_matrix[:,:,1])
    return cloud_matrix
    # np.save(output_filename,cloud_matrix)


def plt_fun(matrix_rp_copy,pred_labels,savepath,issave):
    # matrix_rp_copy=pred_bands.copy()
    rgb_map=np.zeros((matrix_rp_copy.shape[0],matrix_rp_copy.shape[1],3),float)
    rgb_map[:,:,0]= matrix_rp_copy[:,:,8]
    rgb_map[:,:,1]= matrix_rp_copy[:,:,6]
    rgb_map[:,:,2]= matrix_rp_copy[:,:,2]


    Denominator = np.amax(rgb_map)-np.amin(rgb_map) if np.amax(rgb_map)-np.amin(rgb_map)!=0 else 1
    for b in range(3):
        rgb_map[:,:,b]=rgb_map[:,:,b]/Denominator
    f,(ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(rgb_map)
    ax2.imshow(pred_labels)
    if issave==1:
        plt.savefig(savepath)
    else:
        plt.show()
    plt.close()
    return

def unet_img(pretrained_weights = None,input_size = (96,96,9)):
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

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['mean_squared_error'])

    return model

def unet_label(pretrained_weights = None,input_size = (96,96,9),type='random'):
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



def unet_NDWI(pretrained_weights = None,input_size = (96,96,1),type='random'):
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

    if(pretrained_weights):
        model.load_weights(pretrained_weights, by_name=True)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
