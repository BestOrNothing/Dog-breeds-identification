#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:34:48 2018

@author: s1718623
"""

from keras.applications.xception import Xception
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import PIL.Image
from tqdm import tqdm
import sys
from sklearn.cross_validation import train_test_split

img_rows=224
img_cols=224
num_channel=3 # 3 colour channes
batch_size = 32
# Any results you write to the current directory are saved as output.
train_datagen = ImageDataGenerator(
        rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        './unbalanced_data/train',
        target_size=(img_rows,img_cols),
        batch_size=batch_size,
        class_mode='categorical')


valid_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = valid_datagen.flow_from_directory(
        './unbalanced_data/validation',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

# In[12]

net = Xception(include_top=False,
                 input_shape=[img_rows, img_cols, num_channel],
                 pooling='max'
                 )
x = net.output
x = Dense(units=train_generator.num_class,
                activation='softmax')(x)
model = Model(inputs=net.input,outputs=x)
model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit_generator(train_generator, epochs=1,
                    steps_per_epoch=train_generator.n / batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.n / batch_size,
                    verbose=2)