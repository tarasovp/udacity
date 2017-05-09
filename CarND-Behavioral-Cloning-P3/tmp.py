import tensorflow as tf
tf.python.control_flow_ops = tf
import datetime

import os
import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, \
MaxPooling2D, Conv2D, Lambda, Cropping2D, Convolution2D,\
AveragePooling2D
from keras.callbacks import History,TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.cross_validation import train_test_split

from copy import deepcopy
from sklearn.utils import shuffle


from generator import generator
from get_images import get_images

from keras.optimizers import Adam

datadir='/notebooks/udacity/'

datadirs=['~/Desktop/new_training/map1_backward/',
                 '~/Desktop/new_training/map1_forward/',
                 '~/Desktop/new_training/map1_recovery_backward/',
                 '~/Desktop/new_training/map1_recovery_forward/',
                 '~/Desktop/new_training/map2_forward/',
                 '~/Desktop/new_training/map2_backward/',
                 '~/Desktop/new_training/map2_recovery_forward/',
                 '~/Desktop/new_training/map2_recovery_backward/',
                   '~/Desktop/new_training/map1_error_correction/',
                   '~/Desktop/new_training/map2_error_correction/'
         ]

datadirs=[a.replace('~/Desktop',datadir) for a in datadirs]

#datadirs=['~/Desctop/car_data/data/']

images=get_images(datadirs,0.1)
images=images.head(1000)

from car_models import *

names, steering, inverse = images.img.values, images.real.values, images.inverse.values

#names, steering, inverse = shuffle (names, steering, inverse)

def get_images(names, y, batch_size = 32, preprocessing = lambda x:x, inverse=None):
    images = []
    angles = np.zeros((len(names)),dtype=float)
    for now in range(len(names)):

        image = cv2.imread(names[now])
        angle=y[now]
        
        images.append(preprocessing(image))
        angles[now] = y[now]

            
        
    return np.array(images), angles
    
#cnt=10000
    
#names, steering, inverse = names[:cnt], steering[:cnt], inverse[:cnt]

test_set=get_images(names, steering)

#print (test_set[0].shape)
#print (names)


model=nvidia_net()
model_dir='results/'
model.load_weights(model_dir+'model_nvidia_net_02_adam.h5')
t1=datetime.datetime.now()
preds=model.predict(test_set[0])
t2=datetime.datetime.now()

model=modified_vgg()
model.load_weights(model_dir+'model_mod_adam_adam.h5')
t3=datetime.datetime.now()
pred_vgg=model.predict(test_set[0])
t4=datetime.datetime.now()

from sklearn.metrics import mean_squared_error

print ('Vgg time:', (t4-t3).total_seconds(), ' Vgg score:',mean_squared_error(pred_vgg,test_set[1]))
print ('Nvidia time:', (t2-t1).total_seconds(), ' Nvidia score:',mean_squared_error(preds,test_set[1]))

images['pred_nvidia']=preds
images['pred_vgg']=pred_vgg

images.to_csv('images.csv')

#print (mean_squared_error(pred_vgg,test_set[1]), mean_squared_error(preds,test_set[1]), mean_squared_error(preds,pred_vgg))
