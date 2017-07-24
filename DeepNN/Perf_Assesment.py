from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.layers import Input
from keras.models import load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import initializers
from keras.models import Sequential, Model

from keras.applications import imagenet_utils
import numpy as np

import h5py as h5py

#Uncomment if to be used with Python2
#from __future__ import division

base_model = InceptionV3(weights='imagenet',include_top=False,input_shape = (250, 400, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)

#Only for version 2:
x = Dense(1024, activation='relu',kernel_initializer=initializers.VarianceScaling(scale=2.0))(x)

# and a logistic layer
predictions = Dense(1, activation='sigmoid',kernel_initializer=initializers.VarianceScaling(scale=2.0))(x)

model= Model(inputs=base_model.input, outputs=predictions)

model.load_weights("Inceptionv3_2.h5")

#Performance Assesment :
import os

#For Clean Cars :
root_path="../PICS/testnew/"
direc="Clean/"
pred_sum=0
for pic in os.listdir(root_path+direc):
    img = image.load_img(root_path+direc+pic, target_size=(250,400))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = int(round(model.predict(img)))
    pred_sum+=preds
print("False Positive Rate :")
print(pred_sum/len(os.listdir(root_path+direc)))


#For Damaged Cars :
root_path="../PICS/testnew/"
direc="Damaged/"
pred_sum=0
idx=1
for pic in os.listdir(root_path+direc):
    img = image.load_img(root_path+direc+pic, target_size=(250,400))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    preds = int(round(model.predict(img)))
    pred_sum+=preds
    idx+=1
print("Detection Rate :")
print(pred_sum/len(os.listdir(root_path+direc)))
