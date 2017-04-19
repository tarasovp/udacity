import cv2
import numpy as np
from sklearn.utils import shuffle


def generator(names, y, batch_size = 32, preprocessing = lambda x:x, inverse=None):
    """
    Generator of images for NN,
    """


    total_items = len(names)
    curr_item = 0
    if not inverse:
        inverse = np.ones(total_items, dtype=np.uint8)
        
    while (True):
        images = []
        angles = np.zeros((batch_size),dtype=float)
        for j in range(batch_size):
            now=curr_item%total_items
            
            image = cv2.imread(names[now])
            angle=y[now]
            if inverse[now]==-1:
                image=np.fliplr(image)
                angle*=-1
                
            images.append(preprocessing(image))
            angles[j] = y[now]
            
            curr_item +=1
        
        if curr_item>total_items:
            curr_item=curr_item%total_items
            names, y, inverse = shuffle(names, y, inverse)
        
        yield np.array(images), angles