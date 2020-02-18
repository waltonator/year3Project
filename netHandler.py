import tensorflow as tf
import keras

from keras.models import model_from_json

def saveNet(model, dir) :
    jModel = model.to_json()
    with open(dir + ".json", "w") as json_file:
        json_file.write(jModel)
    model.save_weights(dir + ".h5")

def loadNet(dir) :
    jFile = open(dir + ".json", 'r')
    loaded = json_file.read()
    jFile.close()
    mod = model_from_json(loaded)
    mod.load_weights(dir + ".h5")
    return mod
