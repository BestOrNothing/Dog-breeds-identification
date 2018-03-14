drop = False
flip = False
rot = False
scale = False
nb_epochs = 20
# coding: utf-8

# **This kernel is created to show the standard step-by-step process in handling image data. However, given the time limit of an hour, the kernel can only reach a low validation accuracy. Another way  of trainning the model from scratch is to run the script on a very powerful computer or using cloud computing. If you want to save time and computational power, you can also pre-process the data in the same manner and use [ImageNet pre-trained models](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/). Now let's begin and hope you enjoy it. **

# **Libraries that you need for image data preprocessing**

# In[1]:

from util import flip_aug
from util import rot_aug
from util import scale_aug
from util import central_scale_images
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import tensorflow as tf

import sklearn
from sklearn.cross_validation import train_test_split

from keras.models import Sequential  # initial NN
from keras.layers import Dense, Dropout # construct each layer
from keras.layers import Convolution2D # swipe across the image by 1
from keras.layers import MaxPooling2D # swipe across by pool size
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator


img_rows=224
img_cols=224
num_channel=3 # 3 colour channes
# Any results you write to the current directory are saved as output.
train_datagen = ImageDataGenerator(
        rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        './unbalanced_data/train',
        target_size=(img_rows,img_cols),
        batch_size=32,
        class_mode='categorical')


valid_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = valid_datagen.flow_from_directory(
        './unbalanced_data/validation',
        target_size=(img_rows, img_cols),
        batch_size=32,
        class_mode='categorical')

model = Sequential()


# **I have a rather simple CNN here**
# 1. Convetional layer (detect features in image matrix)
# 2. Pooling layer (recongise features in different angle and/or size)
# 3. Convetional layer
# 4. Pooling laye
# 5. Flattening layer (flatten layers in array of imput)
# 6. Full connected layer (full connected ANN)
# 7. Output layer

# In[16]:


# retifier ensure the non-linearity in the processing 
model.add(Convolution2D (filters = 32, kernel_size = (5,5),padding = 'Same', 
                         activation ='relu', input_shape = (img_rows, img_cols, num_channel))) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D (filters = 64, kernel_size = (4,4),padding = 'Same', 
                         activation ='relu')) 
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D (filters = 128, kernel_size = (3,3),padding = 'Same',
                                                  activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
if drop:
    model.add(Dropout(0.5))
model.add(Flatten()) 
# fully connected ANN 
model.add(Dense(units = 500, activation = 'relu'))
if drop:
    model.add(Dropout(0.5))
# output layer
model.add(Dense(units = 120, activation = 'softmax')) 


# **Compile the model**

# In[17]:


model.compile(optimizer = 'adam' , loss = "categorical_crossentropy", metrics=["accuracy"]) 


# **Fit the model into data**

# In[18]:
history = model.fit_generator(train_generator, epochs=10,
                              steps_per_epoch=1000,
                              validation_data=validation_generator,
                              validation_steps=400)

