import cv2
import numpy as np
from sklearn.utils import shuffle


def generator (names, y_data, batch_size = 32,preprocessing = lambda x:x):
    total_items = len(names)
    curr_item = 0
    while (True):
        image_data = []
        steering_data = np.zeros((batch_size),dtype=float)
        for j in range(batch_size):
            image_name = names[curr_item]
            image = cv2.imread(image_name)
            image_data.append(preprocessing(image))
            steering_data[j] = y_data[curr_item]
            curr_item = (curr_item+1)%total_items
        yield np.array(image_data), steering_data


def generator2 (names, y_data, batch_size = 32,preprocessing = lambda x:x, inverse=None):
    total_items = len(names)
    curr_item = 0
    tr=False
    while (True):
        image_data = []
        steering_data = np.zeros((batch_size),dtype=float)
        for j in range(batch_size):
            image_name = names[curr_item]
            image = cv2.imread(image_name)
            #если есть inverse надр развернуть
            if inverse and inverse[curr_item]:
                image_data.append(preprocessing(np.fliplr(image))) 
                steering_data[j] = -y_data[curr_item]
            else:
                image_data.append(preprocessing(image))
                steering_data[j] = y_data[curr_item]
            
            #need to shuffle
            if curr_item>total_items:
                tr=True
            curr_item = (curr_item+1)%total_items
                
        #after each iteration add shuffline
        if tr:
            tr=False
            if inverse:
                names, y_data,inverse = shuffle(names, y_data,inverse)
            else:
                names, y_data = shuffle(names, y_data)
            
        yield np.array(image_data), steering_data