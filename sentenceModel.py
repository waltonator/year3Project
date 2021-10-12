import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Input, Lambda

url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

#Sentence encoder used in original approach
def newSentenceEncoder() :
    tf.compat.v1.disable_eager_execution()
    embed = hub.Module(url)
    def UniversalEmbedding(x) :
        return embed(tf.squeeze(tf.cast(x, tf.string)))

    input = Input(shape=(1,), dtype=tf.string)
    embedding = Lambda(UniversalEmbedding, output_shape=(512, ))(input)

    d1 = Dense(1024, activation= 'relu')(embedding)
    d2 = Dense(2048, activation= 'relu')(d1)
    out = Dense(1024, activation= 'relu')(d2)

    mod = Model(input, out)
    return mod

#Sentence encoder used in new approach, simply consists of universal sentence encoder
def newSentenceEncoderTwo() :
    tf.compat.v1.disable_eager_execution()
    embed = hub.Module(url)
    def UniversalEmbedding(x) :
        return embed(tf.squeeze(tf.cast(x, tf.string)))

    input = Input(shape=(1,), dtype=tf.string)

    embedding = Lambda(UniversalEmbedding, output_shape=(512, ))(input)

    mod = Model(input, embedding)
    return mod
