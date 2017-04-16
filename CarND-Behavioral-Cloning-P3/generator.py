import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle


def generator(df, dataset, correction=0.2, preprocessing=lambda x:x, batch_size=128):
    """
    Generator of images for NN,
    df - dataframe with columns img, side, steering, dataset, inverse
    correction - how much to add for left/right images
    preprocessing - preprocessing function
    and batch_size 
    """
    
    samples=df[df.dataset==dataset].values
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for i, batch_sample in enumerate(batch_samples):
                img,side,steering,dataset,inverse  = batch_sample
                image = cv2.imread(img)
                angle=steering+side*correction
                if inverse:
                    image=np.fliplr(image)
                    angle*=-1

                angles.append(angle)
                images.append(preprocessing(image))



            yield sklearn.utils.shuffle(np.array(images), np.array(angles))