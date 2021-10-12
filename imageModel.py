import tensorflow as tf
import random
import numpy as np
import os

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input, UpSampling2D, Reshape
from tensorflow.keras.models import Model

#original design for auto-encoder, has to many layers
def newImgModel(inSize = 256) :


    inShape = (inSize, inSize, 3)
    reShape = (int(inSize / 4), int(inSize / 4), 3)
    rnum = int(inSize / 4) * int(inSize / 4) * 3

    #encoder
    inputImg = Input(shape = inShape)
    conv1 = Conv2D(64, (3, 3), input_shape = inShape, activation='relu', padding='same')(inputImg)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv7)
    conv8 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv9)
    f = Flatten()(pool4)
    d1 = Dense(2048)(f)
    encoded = Dense(1024, activation='relu')(d1)

    #decoder
    d2 = Dense((rnum), activation= 'relu')(encoded)
    r1 = Reshape(reShape)(d2)
    conv10 = Conv2D(256, (3, 3), activation='relu', padding='same')(r1)
    conv11 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv10)
    up1 = UpSampling2D((2,2))(conv11)
    conv12 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv13 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv12)
    up2 = UpSampling2D((2,2))(conv13)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2)

    autoencoder = Model(inputImg, decoded)
    encoder = Model(inputImg, encoded)

    return autoencoder, encoder

#final auto-encoder used
def newImgModelTwo(inSize = 320) :
    inShape = (inSize, inSize, 3)
    reShape = (int(inSize / 4), int(inSize / 4), 8)
    rnum = int(inSize / 4) * int(inSize / 4) * 8

    #encoder
    inputImg = Input(shape = inShape) #320,320,3
    conv1 = Conv2D(64, (3, 3), input_shape = inShape, activation='relu', padding='same')(inputImg) #320,320,64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #160,160,64
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1) #160,160,128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #80,80,128
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2) #80,80,256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #40,40,256
    f = Flatten()(pool3)

    encoded = Dense(1024, activation='relu')(f)

    #decoder
    d2 = Dense((rnum), activation= 'relu')(encoded)
    r1 = Reshape(reShape)(d2) #80,80,8
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(r1) #80,80,256
    up1 = UpSampling2D((2,2))(conv4) #80,80,256
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1) #160,160,128
    up2 = UpSampling2D((2,2))(conv5) #160,160,128
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up2) #320,320,64

    autoencoder = Model(inputImg, decoded)
    encoder = Model(inputImg, encoded)

    return autoencoder, encoder

    #Image encoder used in the new approach
def newImgModelThree(inSize = 320) :
    inShape = (inSize, inSize, 3)
    rnum = int(inSize / 4) * int(inSize / 4) * 8

    #encoder
    inputImg = Input(shape = inShape) #320,320,3
    conv1 = Conv2D(64, (3, 3), input_shape = inShape, activation='relu', padding='same')(inputImg) #320,320,64
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #160,160,64
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1) #160,160,128
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #80,80,128
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2) #80,80,256
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) #40,40,256
    f = Flatten()(pool3)
    
    encoded = Dense(512, activation='tanh')(f)

    encoder = Model(inputImg, encoded)

    return encoder
