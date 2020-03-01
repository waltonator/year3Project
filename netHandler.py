import tensorflow as tf
import keras

from keras.models import model_from_json
#from tf.keras.models import load_model
#from tensorflow.keras.models import load_model

def saveNet(model, dir) :
    jModel = model.to_json()
    with open(dir + ".json", "w") as json_file:
        json_file.write(jModel)
    model.save_weights(dir + ".h5")

def loadNet(dir) :
    jFile = open(dir + ".json", 'r')
    loaded = jFile.read()
    jFile.close()
    #mod = model_from_json(loaded)
    mod = tf.keras.models.model_from_json(loaded)
    mod.load_weights(dir + ".h5")
    return mod
