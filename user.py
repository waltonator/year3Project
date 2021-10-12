import os
import numpy
import train
import retrieveImage

from retrieveImage import getImages, getSeq

#A simple interface to make using the tools in this project easier
def main() :
    inuse = True
    while (inuse) :
        unsel = True
        print('select option')
        print('1: train new models')
        print('2: retrieve images for input text')
        print('3: get a sequence of images for a sequence of sentences')
        print('4: test image encoder with validation data')
        print('5: exit')
        while (unsel) :
            val = input('Enter a number: ')
            try :
                selection = int(val)
                unsel = False
                if (selection == 1) :
                    ie = input('Train new image encoder?')
                    nie = False
                    cd = input('Create new image encoder dataset?')
                    ncd = False
                    se = input('Create new image vector dictionary?')
                    nse = False
                    if (ie == 'y' or ie == 'yes') :
                        nie = True
                    if (cd == 'y' or cd == 'yes') :
                        ncd = True
                    if (se == 'y' or se == 'yes') :
                        nse = True
                    #train.start(iTrain = nie, dTrain = ncd, sTrain = nse)
                    train.startTwo(iTrain = nie, dTrain = ncd, sTrain = nse) #Start training new models
                if (selection == 2) :
                    n = input('enter number of sentences : ')
                    n = int(n)
                    sents = []
                    for i in range(n) :
                        sents.append(input('Enter the next sentence : '))
                    ti = input('Use training images?')
                    uti = False
                    vi = input('use validation images?')
                    uvi = False
                    tsi = input('use test images?')
                    utsi = False
                    if (ti == 'y' or ti == 'yes') :
                        uti = True
                    if (vi == 'y' or vi == 'yes') :
                        uvi = True
                    if (tsi == 'y' or tsi == 'yes') :
                        utsi = True
                    if (n == 1) :
                        img, _, _ = getImages([sents[0], 'unimportant text'], uti, uvi, utsi) #gets an image for input text
                        print('image id : ' + str((img[0])[0]))
                    else :
                        imgs, _, _ = getImages(sents, uti, uvi, utsi) #gets multiple images for multiple pieces of input text
                        for img in imgs :
                            print('image id : ' + img[0])
                if (selection == 3) :
                    n = input('enter number of sentences : ')
                    n = int(n)
                    sents = []
                    for i in range(n) :
                        sents.append(input('Enter the next sentence : '))
                    ti = input('Use training images?')
                    uti = False
                    vi = input('use validation images?')
                    uvi = False
                    tsi = input('use test images?')
                    utsi = False
                    if (ti == 'y' or ti == 'yes') :
                        uti = True
                    if (vi == 'y' or vi == 'yes') :
                        uvi = True
                    if (tsi == 'y' or tsi == 'yes') :
                        utsi = True
                    if(n == 1) :
                        img, _, _ = getImages([sents[0], 'unimportant text'], uti, uvi, utsi) #A sequence of 1 is just a single image
                        print('image id : ' + str((img[0])[0]))
                    else :
                        imgs = getSeq(sents, uti, uvi, utsi) #returns a sequence of images
                        for img in imgs :
                            print(img)
                if (selection == 4) :
                    train.testValDat()
                if (selection == 5) :
                    print('goodbye')
                    inuse = False
            except ValueError as err:
                print('invalid input')
                print(err)

if __name__ == "__main__":
    main()
