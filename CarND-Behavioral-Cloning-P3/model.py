import tensorflow as tf
tf.python.control_flow_ops = tf

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
from car_models import *



datadirs=['/notebooks/udacity/new_training/map1_backward/',
                 '/notebooks/udacity/new_training/map1_forward/',
                 '/notebooks/udacity/new_training/map1_recovery_backward/',
                 '/notebooks/udacity/new_training/map1_recovery_forward/',
                 '/notebooks/udacity/new_training/map2_forward/',
                 '/notebooks/udacity/new_training/map2_backward/',
                 '/notebooks/udacity/new_training/map2_recovery_forward/',
                 '/notebooks/udacity/new_training/map2_recovery_backward/',
                   '/notebooks/udacity/new_training/map1_error_correction/',
                   '/notebooks/udacity/new_training/map2_error_correction/'
         ]

images=get_images(datadirs,0.1)

#different preprocessing methods
size=(40,80)


names, steering, inverse = images.img.values, images.real.values, images.inverse.values


names_train, names_valid, steering_train, steering_valid, inverse_train, inverse_valid = \
    train_test_split(names, steering, inverse, test_size=0.2)


model=modified_vgg()

checkpoint = ModelCheckpoint('tmp_lap1.h5', monitor='val_mean_squared_error', verbose=1,
                              save_best_only=True, mode='min')
early_stop = EarlyStopping(monitor='val_mean_squared_error',\
                               min_delta=0.001, patience=3,
                                verbose=1, mode='min')

#for validation - only first lap
lap1=[i for i in range(len(names_valid)) if 'map1' in names_valid[i]]
valid_gen_lap1=generator (names_valid[lap1], steering_valid[lap1], batch_size=128)

train_gen=generator(names_train, steering_train, batch_size=128)

tb=TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)

history = History()

model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
 

q=model.fit_generator(train_gen, \
                    samples_per_epoch=len(names_train),\
                    nb_epoch=100,\
                    verbose=1,\
                    validation_data=valid_gen_lap1, \
                    nb_val_samples=len(lap1),\
                    callbacks=[checkpoint, early_stop,history,tb]\
                   )
    
f=open('history.pk1' % name,'wb')
pickle.dump(history.history,f,-1)
f.close()



