# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 20:01:26 2020

@author: Elitebook
"""
# Importing all necessary libraries 
# Importing all necessary libraries 
import cv2 
import os 

# Read the video from specified path 
cam = cv2.VideoCapture("Jurassic_Park.mp4") 

try: 
	
	# creating a folder named data 
	if not os.path.exists('img'): 
		os.makedirs('img') 

# if not created then raise error 
except OSError: 
	print ('Error: Creating directory of data') 

# frame 
currentframe = 0

while(True): 
	
	# reading from frame 
	ret,frame = cam.read() 

	if ret: 
		# if video is still left continue creating images 
		name = './img/frame' + str(currentframe) + '.jpg'
		print ('Creating...' + name) 

		# writing the extracted images 
		cv2.imwrite(name, frame) 

		# increasing counter so that it will 
		# show how many frames are created 
		currentframe += 1
	else: 
		break

# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 

#%%
from tqdm import tqdm
import cv2
import os
import numpy as np
#%%
img_path = 'data_new'



class1_data = []
class2_data = []
class3_data = []
class4_data = []
class5_data = []
for classes in os.listdir(img_path):
        fin_path = os.path.join(img_path, classes)
        for fin_classes in tqdm(os.listdir(fin_path)):
            img = cv2.imread(os.path.join(fin_path, fin_classes))
            img = cv2.resize(img, (224,224))
            img = img/255.
            if classes == 'ian_malcolm':
                class1_data.append(img)
            else:
                class2_data.append(img)

class1_data = np.array(class1_data)
class2_data = np.array(class2_data)
#%%
from keras.applications import VGG16

model = VGG16(include_top=False,input_shape = (224,224,3), weights='imagenet')
#%%
#import tensorflow_addons as tfa
#tqdm_callback = tfa.callbacks.TQDMProgressBar()
vgg_class1 = model.predict(class1_data,verbose = 1)
vgg_class2 = model.predict(class2_data,verbose = 1) 
#%%
from keras.layers import Input, Dense, Dropout,InputLayer
from keras.models import Model
from keras.models import Sequential

inputs = Input(shape=(7*7*512,))
 
dense1 = Dense(1024, activation = 'relu')(inputs)
drop1 = Dropout(0.5)(dense1)
dense2 = Dense(512, activation = 'relu')(drop1)
drop2 = Dropout(0.5)(dense2)
outputs = Dense(1, activation = 'sigmoid')(drop2)
 
model1 = Model(inputs, outputs)
model1.summary()
#%%
train_data = np.concatenate((vgg_class1[:515], vgg_class2[:754]), axis = 0)
train_data = train_data.reshape(train_data.shape[0],7*7*512)

valid_data = np.concatenate((vgg_class1[515:], vgg_class2[754:]), axis = 0)
valid_data = valid_data.reshape(valid_data.shape[0],7*7*512)

#%%
train_label = np.array([0]*vgg_class1[:515].shape[0] + [1]*vgg_class2[:754].shape[0])
valid_label = np.array([0]*vgg_class1[515:].shape[0] + [1]*vgg_class2[754:].shape[0])
#%%
import tensorflow as tf
from keras.callbacks import ModelCheckpoint

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.logging.set_verbosity(tf.logging.ERROR)
model1.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['acc'])

filepath="best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',  verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
#%%
from keras.models import load_model
model1 = load_model("best_model.h5")

#%%
model1.fit(train_data, train_label, epochs = 10, batch_size = 64, validation_data = (valid_data, valid_label), verbose = 1, callbacks = callbacks_list)
#%%
import os
import numpy as np

ian_images = []
no_ian_images = []

test_path = 'img'

for test in tqdm(os.listdir(test_path)):
    test_img = cv2.imread(os.path.join(test_path, test))
    test_img = cv2.resize(test_img, (224,224))
    test_img = test_img/255.
    test_img = np.expand_dims(test_img, 0)
    pred_img = model.predict(test_img)
    pred_feat = pred_img.reshape(1, 7*7*512)
    out_class = model1.predict(pred_feat)
    if out_class < 0.5:
        ian_images.append(out_class)
    else:
        no_ian_images.append(out_class)










