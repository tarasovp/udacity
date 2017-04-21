import tensorflow as tf
tf.python.control_flow_ops = tf

import pandas as pd
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, \
MaxPooling2D, Conv2D, Lambda, Cropping2D, Convolution2D,\
AveragePooling2D,GlobalAveragePooling2D,Input, merge
from keras.callbacks import History,TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.cross_validation import train_test_split
from keras.models import Model

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16


def modified_lenet():
    """
    Lenet, added cropping, normalization and average pooling to make model smaller
    """
    
    model=Sequential()

    model.add(Lambda(lambda x: (x / 255.0) - 0.5,input_shape=(160,320,3)))
    model.add(AveragePooling2D(pool_size=(4,4)))
    model.add(Cropping2D(cropping=((12,4), (0,0))))


    model.add(Convolution2D(32,3, 3,  activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    return model

def nvidia_net(drp=0.5):
    """
    neural network from Nvidia paper
    """
    model=Sequential()
    
    model.add(Cropping2D(cropping=((50,20), (0,0)), \
                         input_shape=(160,320,3)))

    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    
    model.add(Convolution2D(24, 5, 5,  border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Flatten())
    

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(drp))

    model.add(Dense(100, activation='relu'))
    model.add(Dropout(drp))


    model.add(Dense(50, activation='relu'))
    model.add(Dropout(drp))

    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model

def inception():
    """
    inveption V3 with new last layers
    """
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet',input_shape=(160,320,3) , include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)


    x = Dense(116, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(50, activation='relu')(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='tanh')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

def vgg16():
    """
    vgg16 with new last layers
    """
    
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(160,320,3) )

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)


    x = Dense(116, activation='relu')(x)

    x = Dropout(0.5)(x)

    x = Dense(50, activation='relu')(x)

    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(1, activation='tanh')(x)

    # this is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    return model

def modified_vgg():
    #got and a bit modified from https://github.com/Valtgun/ 
    # ----------------------
    # Model - ideas from VG type network
    inp = Input(shape=(160,320,3))


    x=Cropping2D(cropping=((60,20), (0,0)))(inp)

    x=Lambda(lambda x: (x / 255.0) - 0.5)(x)

    x= MaxPooling2D((2,2))(x)
    # First convolution is for model to determine the 'best' colorspace weights
    x = Conv2D(3, 1, 1, border_mode='same', activation='relu')(inp)
    # Reduce dimensions
    x = MaxPooling2D((2,2))(x) #20x80

    # First convolution layer
    x1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(x)
    x1 = Conv2D(32, 3, 3, border_mode='same', activation='relu')(x1)
    x1 = MaxPooling2D((2,2))(x1) #10x40
    x1 = Dropout(0.5)(x1)
    flat1 = Flatten()(x1) # Used for the merge before first fully connected layer

    # Second convolution layer
    x2 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x1)
    x2 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x2)
    x2 = MaxPooling2D((2,2))(x2) #5x20
    x2 = Dropout(0.5)(x2)
    flat2 = Flatten()(x2) # Used for the merge before first fully connected layer

    # Second convolution layer
    x3 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x2)
    x3 = Conv2D(64, 3, 3, border_mode='same', activation='relu')(x3)
    x3 = MaxPooling2D((2,2))(x3) #2x10
    x3 = Dropout(0.5)(x3)
    flat3 = Flatten()(x3) # Used for the merge before first fully connected layer

    # Merge the flattened ouputs after each convolution layer
    x4 = merge([flat1, flat2, flat3], mode='concat')
    # Fully connected layers
    x5 = Dense(512, activation='relu')(x4)
    x6 = Dense(128, activation='relu')(x5)
    x7 = Dense(16, activation='relu')(x6)
    out = Dense(1, activation='linear')(x7)

    model = Model(input=inp, output=out)
    #model.summary()
    return model




