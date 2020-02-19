import tensorflow as tf
import random
import numpy as np
import os

#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, UpSampling2D, Reshape
from tensorflow.keras.models import Model

def newImgModel(inSize) :

    #encoder
    inShape = (inSize, inSize, 3)
    reShape = (int(inSize / 4), int(inSize / 4), 3)
    inputImg = Input(shape = inShape)
    #net['input'] = InputLayer((None, 3, 224, 224))
    conv1 = Conv2D(64, (3, 3), input_shape = inShape, activation='relu', padding='same')(inputImg)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    #net['conv1_1'] = ConvLayer(net['input'], 64, 3, pad=1)
    #net['conv1_2'] = ConvLayer(net['conv1_1'], 64, 3, pad=1)
    #net['pool1'] = PoolLayer(net['conv1_2'], 2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #net['conv2_1'] = ConvLayer(net['pool1'], 128, 3, pad=1)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    #net['conv2_2'] = ConvLayer(net['conv2_1'], 128, 3, pad=1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    #net['pool2'] = PoolLayer(net['conv2_2'], 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #net['conv3_1'] = ConvLayer(net['pool2'], 256, 3, pad=1)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    #net['conv3_2'] = ConvLayer(net['conv3_1'], 256, 3, pad=1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    #net['conv3_3'] = ConvLayer(net['conv3_2'], 256, 3, pad=1)
    #conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    #net['conv3_4'] = ConvLayer(net['conv3_3'], 256, 3, pad=1)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    #net['pool3'] = PoolLayer(net['conv3_4'], 2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    #net['conv4_1'] = ConvLayer(net['pool3'], 512, 3, pad=1)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    #net['conv4_2'] = ConvLayer(net['conv4_1'], 512, 3, pad=1)
    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv8)
    #net['conv4_3'] = ConvLayer(net['conv4_2'], 512, 3, pad=1)
    #conv10 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv9)
    #net['conv4_4'] = ConvLayer(net['conv4_3'], 512, 3, pad=1)
    #conv12 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv11)
    #net['pool4'] = PoolLayer(net['conv4_4'], 2)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
    #net['conv5_1'] = ConvLayer(net['pool4'], 512, 3, pad=1)
    #conv13 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    #net['conv5_2'] = ConvLayer(net['conv5_1'], 512, 3, pad=1)
    #conv14 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv13)
    #net['conv5_3'] = ConvLayer(net['conv5_2'], 512, 3, pad=1)
    #conv15 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv14)
    #net['conv5_4'] = ConvLayer(net['conv5_3'], 512, 3, pad=1)
    #conv16 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv15)
    #net['pool5'] = PoolLayer(net['conv5_4'], 2)
    #pool5 = MaxPooling2D(pool_size=(2, 2))(conv16)
    f = Flatten()(pool4)
    #net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    d1 = Dense(2048)(f)
    #net['fc7'] = DenseLayer(net['fc6'], num_units=4096)
    #d2 = Dense(2048)(d1)
    #net['fc8'] = DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
    encoded = Dense(1024, activation='softmax')(d1)
    #net['prob'] = NonlinearityLayer(net['fc8'], softmax)
    #model.add(Activation('softmax'))

    #decoder
    #d3 = Dense(1024, activation= 'relu')(encoded)
    #d4 = Dense(2048, activation= 'relu')(d3)
    d5 = Dense((12288), activation= 'relu')(encoded)
    r1 = Reshape(reShape)(d5)
    #conv17 = Conv2D(512, (3, 3), activation='relu', padding='same')(r1)
    #conv18 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv17)
    #up1 = UpSampling2D((2,2))(conv18)
    conv19 = Conv2D(256, (3, 3), activation='relu', padding='same')(r1)
    conv20 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv19)
    up2 = UpSampling2D((2,2))(conv20)
    conv21 = Conv2D(128, (3, 3), activation='relu', padding='same')(up2)
    conv22 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv21)
    up3 = UpSampling2D((2,2))(conv22)
    decoded = Conv2D(3, (3, 3), activation='relu', padding='same')(up3)

    autoencoder = Model(inputImg, decoded)
    encoder = Model(inputImg, encoded)

    return autoencoder, encoder
