import numpy as np
import tensorflow as tf
import glob
import random
from sklearn import model_selection

from keras.applications.nasnet import NASNetLarge
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Activation
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
from itertools import chain, repeat, cycle
import keras
from sklearn.metrics import f1_score
import cv2

import os
import shutil

def trainClasses():
    trainClasses = set()
    with open("trainclasses.txt") as f:
        for line in f:
            trainClasses.add(line.strip())

    idxToClass = {}
    with open("classes.txt") as f:
        for line in f:
            idx, cname = line.strip().split("\t")
            idxToClass[int(idx)] = cname
    
    print(idxToClass)
    
    classToY = {}
    with open("predicate-matrix-binary.txt") as f:
        for idx, line in enumerate(f):
            cname = idxToClass[idx+1]
            if cname in trainClasses:
                classToY[cname] = list(map(int, line.strip().split()))
    
    print(classToY)


    X = []
    Y = []
    for className in trainClasses:
        yvec = classToY[className]
        DATA_DIR = './animalPics/' + className
        for filename in os.listdir(DATA_DIR):
            print(filename)
            with open(os.path.join(DATA_DIR, filename), 'r') as f:
                npimg = np.fromfile(f, np.uint8)
                X.append(cv2.imdecode(npimg, cv2.IMREAD_COLOR))
                Y.append(yvec)
    
    X = np.array(X)
    Y = np.array(Y)

    print(X.shape)
    print(Y.shape)
    
    return X, Y

def identifyFeatures():
    pass

def classifyAttributes():
    pass

def zeroShot():
    pass

trainClasses()
