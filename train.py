import tensorflow as tf
import random
import keras
import numpy as np
import os
import json
import cv2

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from imageModel import newImgModel
from sentenceModel import newSentenceEncoder
from netHandler import saveNet, loadNet




def start(imgSize = 320, iTrain = True, sTrain = True) :

    print('Loading dataset')

    trainDat = json.load(open('../DataSets/VIST/dii/train.description-in-isolation.JSON'))
    valDat = json.load(open('../DataSets/VIST/dii/val.description-in-isolation.JSON'))

    numTrainImages = 0
    numValImages = 0

    trainDir = 'C:/Users/walto/Documents/Uni/Project/DataSets/VIST/training'
    valDir = 'C:/Users/walto/Documents/Uni/Project/DataSets/VIST/validation'

    #imgTData = constructDataSet(trainDat, trainDir, imgSize)
    imgTAds = imageSet(trainDat, trainDir, numTrainImages)
    #imgVData = constructDataSet(valDat, valDir, imgSize)
    imgVAds = imageSet(valDat, valDir, numValImages)

    #inSize = (imgSize, imgSize, 3)

    #print('Building data generators')



    imgEnc = iEncoder(iTrain, imgSize, imgTAds, imgVAds, numTrainImages, numValImages, trainDir, valDir)

    ##sntcModel = sMod(sTrain)

    #print('training model')

    #autoencoder.fit(imgTData, imgTData, epochs=5, batch_size=64, validation_data=(imgVData, imgVData))


    #[{'original_text': 'A group of people standing at a bar smiling.', 'album_id': '72157624875976415', 'photo_flickr_id': '5009759189', 'photo_order_in_story': 4, 'worker_id': 'S8235PXB9Z3JZYL', 'text': 'a group of people standing at a bar smiling .', 'tier': 'descriptions-in-isolation'}]]
    #photo file name is photo_flickr_id

    #for item in trainDat

def iEncoder(buildNew, imgSize, imgTAds, imgVAds, numTrainImages, numValImages, tDir, vDir, epochs = 5, batchSize = 8) :
    if (buildNew) :
        autoTrainDataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2)
        autoTrainGen = autoTrainDataGen.flow_from_directory(tDir, target_size=(imgSize, imgSize), batch_size=batchSize, class_mode='input')
        autoValGen = autoTrainDataGen.flow_from_directory(vDir, target_size=(imgSize, imgSize), batch_size=batchSize, class_mode='input')

        print('Building models')
        inSize = (imgSize, imgSize, 3)
        print('training image encoder')
        auto, enc = newImgModel(imgSize)
        auto.compile(optimizer='adadelta', loss='binary_crossentropy')
        auto.fit_generator(autoTrainGen, epochs=epochs, validation_data=autoValGen)
        saveNet(enc, './Models/imgEnc')
        return enc
    else :
        return loadNet('./Models/imgEnc')

def sMod(buildNew) :
    if (buildNew) :
        return newSentenceEncoder()
    else :
        return loadNet('./Models/sntMod')

def imageSet(data, dir, counter) :
    imgData = []
    data = data.get('annotations')
    for i in data :
        for item in i:
            imgData.append(dir + '/' + item.get('photo_flickr_id'))
            counter += 1
            #print(item.get('photo_flickr_id'))
            #image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.jpg')
            #if image is None :
        #        image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.png')
        #    image = cv2.resize(image, (s, s))
        #    image = img_to_array(image)
        #    imgData.append(image)
    #imgData = np.array(imgData, dtype="float") / 255.0

    return imgData

def constructDataSet(data, dir, s, counter) :
    imgData = []
    data = data.get('annotations')
    for i in data :
        for item in i:
            print(item.get('photo_flickr_id'))
            image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.jpg')
            if image is None :
                image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.png')
            image = cv2.resize(image, (s, s))
            image = img_to_array(image)
            imgData.append(image)
    imgData = np.array(imgData, dtype="float") / 255.0

    return imgData
