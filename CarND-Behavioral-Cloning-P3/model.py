import tensorflow as tf
tf.python.control_flow_ops = tf

import os

os.environ['CUDA_VISIBLE_DEVICES']=str(0)

import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential, Model
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Reshape, merge, Input
from keras.optimizers import Adam
from keras_tqdm import TQDMNotebookCallback
from generator import generator,generator2
from car_models import *
from sklearn.utils import shuffle
from tqdm import tqdm_notebook




from get_images import get_images

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

images=get_images(datadirs,0.08)
image_names_full, y_data_full = images.img.values, images.real.values

#preprocessing function
def proc_img(img): # input is 160x320x3
    img = img[59:138:2, 0:-1:2, :] # select vertical region and take each second pixel to reduce image dimensions
    img = (img / 127.5) - 1.0 # normalize colors from 0-255 to -1.0 to 1.0
    return img # return 40x160x3 image

#generating train/valid sets
names_train, names_valid, y_train, y_valid = train_test_split(image_names_full, y_data_full, test_size=0.02,\
                            random_state=0)

#for each image adding inverse image in train set
inverse_train=[0 for i in y_train]+[1 for i in y_train]


names_train=  np.concatenate([names_train,names_train])
y_train    =   np.concatenate([y_train,y_train])

names_train,y_train,inverse_train=shuffle(names_train,y_train,inverse_train) 



#generators
train_gen=generator2(names_train, y_train, batch_size=64, preprocessing=proc_img, inverse=inverse_train)
valid_gen=generator(names_valid, y_valid, batch_size=64, preprocessing=proc_img)


tb=TensorBoard(log_dir='logs', histogram_freq=1, write_graph=True, write_images=True)

history = History()

model=modified_vgg()
model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['mean_squared_error'])
 
#fitting model
hidtory=model.fit_generator(train_gen, \
                    samples_per_epoch=len(names_train),\
                    nb_epoch=10,\
                    verbose=True,\
                    validation_data=valid_gen, \
                    nb_val_samples=len(names_valid),\
                    callbacks=[tb]\
                   )


    
f=open('history.pk1','wb')
pickle.dump(history.history,f,-1)
f.close()



