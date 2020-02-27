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
from PIL import ImageFile




def start(imgSize = 256, iTrain = True, dTrain = True, sTrain = False) :
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print('Loading dataset')

    trainDat = json.load(open('../DataSets/VIST/dii/train.description-in-isolation.json'))
    valDat = json.load(open('../DataSets/VIST/dii/val.description-in-isolation.json'))

    numTrainImages = 3781
    numValImages = 749

    #trainDir = 'C:/Users/walto/Documents/Uni/Project/DataSets/VIST/training'
    #valDir = 'C:/Users/walto/Documents/Uni/Project/DataSets/VIST/validation'
    trainDir = '../DataSets/VIST/miniTraining'
    valDir = '../DataSets/VIST/miniValidation'


    imgTAds = imageSet(trainDat, trainDir, numTrainImages)
    imgVAds = imageSet(valDat, valDir, numValImages)

    #inSize = (imgSize, imgSize, 3)

    #print('Building data generators')

    print('Building models')

    imgEnc = iEncoder(iTrain, imgSize, imgTAds, imgVAds, numTrainImages, numValImages, trainDir, valDir)

    xt, yt = constructDataSet(trainDat, trainDir, imgSize, imgEnc, Dtrain, "train")
    xv, yv = constructDataSet(valDat, valDir, imgSize, imgEnc, Dtrain, "val")

    sntcModel = sMod(sTrain, xt, yt, xv, yv)

    #print('training model')

    #autoencoder.fit(imgTData, imgTData, epochs=5, batch_size=64, validation_data=(imgVData, imgVData))


    #[{'original_text': 'A group of people standing at a bar smiling.', 'album_id': '72157624875976415', 'photo_flickr_id': '5009759189', 'photo_order_in_story': 4, 'worker_id': 'S8235PXB9Z3JZYL', 'text': 'a group of people standing at a bar smiling .', 'tier': 'descriptions-in-isolation'}]]
    #photo file name is photo_flickr_id

    #for item in trainDat

def iEncoder(buildNew, imgSize, imgTAds, imgVAds, numTrainImages, numValImages, tDir, vDir, epochs = 25, batchSize = 64) :
    if (buildNew) :
        autoTrainDataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2)
        autoTrainGen = autoTrainDataGen.flow_from_directory(tDir, target_size=(imgSize, imgSize), batch_size=batchSize, class_mode='input')
        autoValGen = autoTrainDataGen.flow_from_directory(vDir, target_size=(imgSize, imgSize), batch_size=batchSize, class_mode='input')

        inSize = (imgSize, imgSize, 3)
        print('training image encoder')
        auto, enc = newImgModel(imgSize)
        auto.compile(optimizer='adadelta', loss='mean_squared_error')
        auto.fit_generator(autoTrainGen, epochs=epochs, validation_data=autoValGen)
        saveNet(enc, './Models/imgEnc')
        return enc
    else :
        return loadNet('./Models/imgEnc')

def sMod(buildNew, xt, yt, xv, yv, bs = 64) :
    if (buildNew) :
        print('training sentence encoder')
        se = newSentenceEncoder()
        se.compile(optimizer='adadelta', loss='mean_squared_error')
        se.fit(xDat, yDat, batch_size= bs, epochs= 25, validation_data=(xv, yv))
        saveNet(se, './Models/sntMod')
        return se
    else :
        return loadNet('./Models/sntMod')

def imageSet(data, dir, counter) :
    imgData = []
    data = data.get('annotations')
    for i in data :
        for item in i:
            imgData.append(dir + '/' + item.get('photo_flickr_id'))
            #counter += 1
            #print(item.get('photo_flickr_id'))
            #image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.jpg')
            #if image is None :
        #        image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.png')
        #    image = cv2.resize(image, (s, s))
        #    image = img_to_array(image)
        #    imgData.append(image)
    #imgData = np.array(imgData, dtype="float") / 255.0

    return imgData

def constructDataSet(data, dir, s, model, buildNew, setName) :
    xString = "./Models/" + setName + "x.npy"
    yString = "./Models/" + setName + "y.npy"
    if (buildNew) :
        print("Constructing dataset")
        xDat, yDat = [], []
        data = data.get('annotations')
        for i in data :
            for item in i:
                #print(item.get('photo_flickr_id'))
                image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.jpg')
                if image is None :
                    image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.png')
                if (image is not None) :
                    image = cv2.resize(image, (s, s))
                    image = img_to_array(image)
                    image = [image]
                    image = np.array(image, dtype="float") / 255.0
                    rep = model.predict(image)
                    xDat.append(item.get('text'))
                    yDat.append(rep[0])
                    #xDat.append(image)
        xDat, yDat = np.array(xDat), np.array(yDat, dtype="float")
        np.save(xString, xDat)
        np.save(yString, yDat)
        return xDat, yDat
    else :
        return np.load(xString), np.load(yString)
