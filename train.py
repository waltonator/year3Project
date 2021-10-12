import tensorflow as tf
import random
import keras
import numpy as np
import os
import json
import cv2
import keras.backend as K

from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.callbacks import EarlyStopping
from imageModel import newImgModel, newImgModelTwo, newImgModelThree
from sentenceModel import newSentenceEncoder, newSentenceEncoderTwo
from netHandler import saveNet, loadNet
from PIL import ImageFile

#This trains the models using the new approach
def startTwo(imgSize = 320, iTrain = True, dTrain = False, sTrain = False) :
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print('Loading dataset')
    trainDir = '../DataSets/VIST/miniTraining/train'
    valDir = '../DataSets/VIST/miniValidation/val'

    trainDat = None
    valDat = None
    testDat = None
    if(dTrain or sTrain) :
        trainDat = json.load(open('../DataSets/VIST/dii/train.description-in-isolation.json'))
        valDat = json.load(open('../DataSets/VIST/dii/val.description-in-isolation.json'))
        testDat = json.load(open('../DataSets/VIST/dii/test.description-in-isolation.json'))

    sntcModel = sModTwo(dTrain)
    xt, yt = constructDataSetTwo(trainDat, trainDir, imgSize, sntcModel, dTrain, "train")
    xv, yv = constructDataSetTwo(valDat, valDir, imgSize, sntcModel, dTrain, "val")
    #xv, yv = None, None

    imgEnc = iEncoderTwo(iTrain, imgSize, xt, yt, xv, yv, sTrain)

    trainDir = '../DataSets/VIST/training/train'
    valDir = '../DataSets/VIST/validation/val'
    testDir = '../DataSets/VIST/testing/test'

    constructDic(trainDat, trainDir, imgSize, imgEnc, sTrain, "train")
    constructDic(valDat, valDir, imgSize, imgEnc, sTrain, "val")
    constructDic(testDat, testDir, imgSize, imgEnc, sTrain, "test")

#This trains the models using the initial approach
def start(imgSize = 320, iTrain = True, dTrain = False, sTrain = False) :
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    print('Loading dataset')

    numTrainImages = 3746
    numValImages = 201

    trainDir = '../DataSets/VIST/miniTraining'
    valDir = '../DataSets/VIST/miniValidation'

    print('Building models')

    imgEnc = iEncoder(iTrain, imgSize, numTrainImages, numValImages, trainDir, valDir, dTrain)

    trainDir = '../DataSets/VIST/training/train'
    valDir = '../DataSets/VIST/validation/val'

    trainDat = json.load(open('../DataSets/VIST/dii/train.description-in-isolation.json'))
    valDat = json.load(open('../DataSets/VIST/dii/val.description-in-isolation.json'))

    xt, yt = constructDataSet(trainDat, trainDir, imgSize, imgEnc, dTrain, "train")
    xv, yv = constructDataSet(valDat, valDir, imgSize, imgEnc, dTrain, "val")

    sntcModel = sMod(sTrain, xt, yt, xv, yv)

#tests the image encoder with the validation data
def testValDat() :
    imgEnc = loadNet('./Models/imgEnc')
    xv, yv = np.load('./Models/valx.npy'), np.load('./Models/valy.npy')
    imgEnc.compile(optimizer='adadelta', loss='mean_squared_error')
    xv = np.squeeze(xv)
    imgEnc.evaluate(xv, yv)

#Gets the image encoder for the new approach
def iEncoderTwo(buildNew, imgSize, trX, trY, vX, vY, required, epch = 150, batchSize = 16) :
    if (buildNew) :
        trX = np.squeeze(trX)
        print('training image encoder')
        enc = newImgModelThree()
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        enc.compile(optimizer='adadelta', loss='mean_squared_error')
        enc.fit(trX, trY, batch_size=batchSize, epochs=epch, callbacks=[es], validation_data=(vX, vY))

#use these for dcs machines to save space
        #os.remove("./Models/trainx.npy")
        #os.remove("./Models/trainy.npy")
        #os.remove("./Models/valx.npy")
        #os.remove("./Models/valy.npy")

        saveNet(enc, './Models/imgEnc')
    elif(required) :
        return loadNet('./Models/imgEnc')
    else :
         return None

#Gets the image encoder for initial approach
def iEncoder(buildNew, imgSize, numTrainImages, numValImages, tDir, vDir, required, epochs = 150, batchSize = 16) :
    if (buildNew) :
        autoTrainDataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2)
        autoTrainGen = autoTrainDataGen.flow_from_directory(tDir, target_size=(imgSize, imgSize), batch_size=batchSize, class_mode='input')
        autoValGen = autoTrainDataGen.flow_from_directory(vDir, target_size=(imgSize, imgSize), batch_size=batchSize, class_mode='input')

        print('training image encoder')
        #auto, enc = newImgModel() #for 256
        auto, enc = newImgModelTwo() #for 320
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        auto.compile(optimizer='adadelta', loss='mean_squared_error')
        auto.fit_generator(autoTrainGen, epochs=epochs, validation_data=autoValGen, callbacks=[es])
        saveNet(enc, './Models/imgEnc')
        return enc
    elif (required) :
        return loadNet('./Models/imgEnc')
    else :
         return None

#Sentence encoder for initial approach
def sMod(buildNew, xt, yt, xv, yv, bs = 64, eps = 50) :
    if (buildNew) :
        print('training sentence encoder')

        se = newSentenceEncoder()

        se.compile(optimizer='adadelta', loss='mean_squared_error')

        with tf.compat.v1.Session() as session:

            tf.compat.v1.keras.backend.set_session(session)
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(tf.compat.v1.tables_initializer())
            #history = model.fit(x_train, y_train, epochs=1, batch_size=32)
            se.fit(xt, yt, batch_size= bs, epochs= eps, validation_data=(xv, yv))

            se.save_weights("./Models/sntMod.h5")
        return se
    else :
        se = newSentenceEncoder()
        se.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mean_squared_error'])
        se.evaluate(xv,yv)
        se.load_weights("./Models/sntMod.h5")
        return se

#Sentence encoder for new approach
def sModTwo(required) :
    if(required) :
        print('getting sentence encoder')
        se = newSentenceEncoderTwo()
        se.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['mean_squared_error'])
        print('returning sentence encoder')
        return se
    else :
        return None

#Constructs dataset used in the new approach
def constructDataSetTwo(data, dir, s, model, buildNew, setName) :
    xString = "./Models/" + setName + "x.npy"
    yString = "./Models/" + setName + "y.npy"
    if (buildNew) :
        print("Constructing dataset")
        xDat, yDat, txt = [], [], []
        data = data.get('annotations')
        for i in data :
            for item in i:
                image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.jpg')
                if image is None :
                    image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.png')
                if (image is not None) :
                    image = cv2.resize(image, (s, s))
                    image = img_to_array(image)
                    image = [image]
                    image = np.array(image, dtype="float") / 255.0
                    xDat.append(image)
                    txt.append(item.get('photo_flickr_id'))
        with tf.compat.v1.Session() as session:
            print("getting correct output for dataset")
            tf.compat.v1.keras.backend.set_session(session)
            session.run(tf.compat.v1.global_variables_initializer())
            session.run(tf.compat.v1.tables_initializer())
            yDat = model.predict(np.array(txt))
        print("saving datasets")
        np.save(xString, xDat)
        np.save(yString, yDat)
        return xDat, yDat
    else :
        return np.load(xString), np.load(yString)

#Constructs datasets used in the initial approach
def constructDataSet(data, dir, s, model, buildNew, setName) :
    xString = "./Models/" + setName + "x.npy"
    yString = "./Models/" + setName + "y.npy"
    dString = "./Models/" + setName + "dic.npy"
    dic = {}
    if (buildNew) :
        print("Constructing dataset")
        xDat, yDat = [], []
        data = data.get('annotations')
        for i in data :
            for item in i:
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
                    dic.update({str(item.get('photo_flickr_id')) : rep[0]})
        xDat, yDat = np.array(xDat), np.array(yDat, dtype="float")
        np.save(xString, xDat)
        np.save(yString, yDat)
        np.save(dString, dic)
        return xDat, yDat
    else :
        return np.load(xString), np.load(yString)

#Creates dictionarys of images to vector representation
def constructDic(data, dir, s, model, buildNew, setName) :
    dString = "./Models/" + setName + "dic.npy"
    dic = {}
    if (buildNew) :
        print("Constructing dataset")
        data = data.get('annotations')
        for i in data :
            for item in i:
                image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.jpg')
                if image is None :
                    image = cv2.imread(dir + '/' + str(item.get('photo_flickr_id')) + '.png')
                if (image is not None) :
                    print('img : ' + item.get('photo_flickr_id'))
                    image = cv2.resize(image, (s, s))
                    image = img_to_array(image)
                    image = [image]
                    image = np.array(image, dtype="float") / 255.0
                    rep = model.predict(image)
                    dic.update({str(item.get('photo_flickr_id')) : rep[0]})
        np.save(dString, dic)
