import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def preprocessing():
    im_width = 512
    im_height = 512
    border = 5
    ids = next(os.walk("/content/drive/My Drive/images1/b"))[2]
    print("No. of images = ", len(ids))
    print(ids)
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        img = load_img("/content/drive/My Drive/images1/b/"+id_,grayscale=True)
        x_img = img_to_array(img)
        mask = img_to_array(load_img("/content/drive/My Drive/images1/b. ann/"+"mask_"+id_,   grayscale=True))
        X[n] = x_img/255.0
        y[n] = mask/255.0
    return X,y 
