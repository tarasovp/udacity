import pandas as pd
import os
from copy import deepcopy

def mod_path(path, folder):
    """
    modify path of images to real path
    """
    return folder+'IMG/'+path.split('/')[-1]

def get_pd(folder):
    """
    get images from driving_log from folder
    """
    df=pd.read_csv(folder+'driving_log.csv', skiprows=1,\
            names=['center','left','right','steering','trottle','brake','speed'])
    
    df['center']=df.center.apply(lambda x:mod_path(x,folder))
    df['left']=df.left.apply(lambda x:mod_path(x,folder))
    df['right']=df.right.apply(lambda x:mod_path(x,folder))
    
    return df

def get_images(datadirs):
    """
    create (or read saved) dataframe with imageurls and parametrs
    """
    if not os.path.exists('images.pk1'):
        images=pd.concat([get_pd(a) for a in datadirs])

        train, test_and_valid=train_test_split(range(len(images)),test_size=0.2)
        test,valid=train_test_split(test_and_valid, test_size=0.5)

        s=np.empty(len(images), dtype=np.uint8)
        s[train]=0
        s[valid]=1
        s[test]=2

        images['dataset']=s
        images['inverse']=False

        images2=deepcopy(images)
        images2['inverse']=True
        images2['speed']=images2.speed.apply(lambda x:-x)

        images=pd.concat([images,images2])

        left=deepcopy(images)
        right=deepcopy(images)
        left['side']=1
        right['side']=-1
        left['img']=left.left
        right['img']=right.right
        images['img']=images.center
        images['side']=0

        images=pd.concat([images[['img','side','steering','dataset','inverse']],\
                          left[['img','side','steering','dataset','inverse']],\
                          right[['img','side','steering','dataset','inverse']]\
                         ])


        images.to_pickle('images.pk1')
    else:
        images=pd.read_pickle('images.pk1')
        
    return images