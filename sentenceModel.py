import tensorflow as tf
import tensorflow_hub as hub
#import keras
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Lambda

url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

def newSentenceEncoder() :
    embed = hub.Module(url)

    #model = Sequential()
    def UniversalEmbedding(x) :
        return embed(tf.squeeze(tf.cast(x, tf.string)))

    #model.add(Input(shape=(1,), dtype=tf.string))
    input = Input(shape=(1,), dtype=tf.string)
    #model.add(Lambda(UniversalEmbedding(embed), output_shape=(512, )))
    embedding = Lambda(UniversalEmbedding, output_shape=(512, ))(input)
    #model.add(Activation('relu'))
    #model.add(Dense(2048))
    d1 = Dense(2048, activation= 'relu')(embedding)
    #model.add(Dense(2048))
    d2 = Dense(2048, activation= 'relu')(d1)
    #model.add(Dense(2048))
    d3 = Dense(2048, activation= 'relu')(d2)
    #model.add(Dense(1024))
    #model.add(Activation('softmax'))
    out = Dense(1024, activation= 'softmax')(d3)

    mod = Model(input, out)
    return mod
