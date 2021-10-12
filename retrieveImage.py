import numpy as np
import keras
import tensorflow as tf
import heapq as hq

import netHandler
import sentenceModel

from sentenceModel import newSentenceEncoderTwo
from netHandler import saveNet, loadNet

#given a set of text, get possible images for that piece of text
def getImages(inText, useTrain, useVal, useTest, num = 1) :
    #sntEnc = loadNet('./Models/sntMod')
    se = newSentenceEncoderTwo()
    se.compile(optimizer='adadelta', loss='mean_squared_error')

    trDic = np.load("./Models/traindic.npy", allow_pickle=True).flat[0]
    valDic = np.load("./Models/valdic.npy", allow_pickle=True).flat[0]
    testDic = np.load("./Models/testdic.npy", allow_pickle=True).flat[0]
    search = {}
    if (useTrain) :
        search.update(trDic)
    if (useVal) :
        search.update(valDic)
    if(useTest) :
        search.update(testDic)

    with tf.compat.v1.Session() as session:
        #K.
        tf.compat.v1.keras.backend.set_session(session)
        session.run(tf.compat.v1.global_variables_initializer())
        session.run(tf.compat.v1.tables_initializer())

        print('predicting images')

        posI = se.predict(np.array(inText))

        results = []
        vecs = []
        print('searching images')
        for p in posI :
            srch = search.copy()
            r, v = [], []
            for i in range(num) :
                k, vec = searchDic(srch, p)
                r.append(k)
                v.append(vec)
                srch.pop(k, None)
            results.append(r)
            vecs.append(v)
        return results, vecs, posI

#searches the given dictionary for the vector representation closest to the given vector
def searchDic(dict, vec) :
    k = list(dict.keys())
    v = np.array(list(dict.values()))

    idx = np.sum(np.power((v - vec), 2), axis=1).argmin() #gets the square of the distances and finds the location of the lowest one

    return k[idx], v[idx]

#Returns a sequence of images for a sequence of text
def getSeq(sents, useTrain, useVal, useTest, num = 10) :
    ks, vecs, svecs = getImages(sents, useTrain, useVal, useTest, num = 10)
    imgNum = len(sents)

    out = []
    layers = []
    strt = node(None, 'start')

    print('constructing possible sequences')
    for l in range(imgNum) :
        kys = ks[l]
        vs = vecs[l]
        layer = []
        for i in range(num) :
            cur = node(vs[i], kys[i])
            layer.append(cur)
        layers.append(layer)
    for s in layers[0] :
        strt.succs.append(s)
        strt.sucDis.append(0)
    for m in range(imgNum - 1) :
        nodesA, nodesB = layers[m], layers[m + 1]
        for a in nodesA :
            for b in nodesB :
                h = findFlow(a.vector, b.vector) + findCor(a.vector, b.vector) + findSim(svecs[m], b.vector)
                a.succs.append(b)
                a.sucDis.append(h)
    end = node(None, 'end')
    for n in layers[imgNum - 1] :
        n.succs.append(end)
        n.sucDis.append(0)
    toExp = []
    cur, dist, pred = strt, 0, None
    print('searching possible sequences')
    while(cur.id != 'end') :
        if (not cur.exp) :
            cur.exp = True
            cur.pred = pred
            for s in range(len(cur.succs)):
                nd = cur.sucDis[s] + dist
                hq.heappush(toExp, [nd, cur.succs[s], cur])
        dist, cur, pred = hq.heappop(toExp)
    end.exp = True
    end.pred = pred
    cur = pred
    print('reconstructing sequence')
    while(cur.id != 'start') :
        out.append(cur.id)
        cur = cur.pred
    out.reverse()
    return out

#Returns the flow between 2 vectors, if there the same vector return a large value so that it won't be picked to follow itself
def findFlow(vecA, vecB) :
    cor = findCor(vecA, vecB)
    if(cor == 0) :
        return 1000000
    else :
        return (1 / cor)

#Returns the coherency of 2 vectors
def findCor(vecA, vecB) :
    return np.linalg.norm(vecA - vecB)

#Returns the similarity between 2 vectors
def findSim(vecA, vecB) :
    return np.dot(vecA, vecB)/(np.linalg.norm(vecA)*np.linalg.norm(vecB))

#node class used in uniform-cost search which is used in sequence search
class node:
    def __init__(self, vec, id):
        self.vector = vec
        self.id = id
        self.succs = []
        self.sucDis = []
        self.exp = False
        self.pred = None
    def __lt__(self,other): #needed to allow for use in heap queue
        return self.id < other.id
