
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import pandas as pd
import numpy as np
import PIL.Image
from tqdm import tqdm
import sys
from sklearn.cross_validation import train_test_split


# In[5]:


img_num = int(sys.argv[1])
epochs = int(sys.argv[2])
#img_num = 100
labels = pd.read_csv('./labels.csv')[0:img_num]
breed_count = labels['breed'].value_counts()
print(breed_count.head())
print(breed_count.shape)


# In[6]:


targets = pd.Series(labels['breed'])
one_hot = pd.get_dummies(targets, sparse=True)
one_hot_labels = np.asarray(one_hot)


# In[7]:


img_rows = 224
img_cols = 224
num_channel = 3


# In[8]:


X = []
y = []
i = 0
for f, img in tqdm(labels.values):
    train_img = PIL.Image.open('./train/{}.jpg'.format(f))
    label = one_hot_labels[i]
    train_img = train_img.resize([img_rows, img_cols])
    train_img = np.array(train_img)
    train_img = np.clip(train_img / 255, 0., 1.)
    X.append(train_img)
    y.append(label)
    i += 1


# In[9]:


X = np.array(X, np.float32)


# In[10]:


y = np.array(y, np.uint8)


# In[11]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)


# In[12]:


#AlexNet with batch normalization in Keras 
#input image is 224x224

model = Sequential()
model.add(Convolution2D(64, [11, 11], padding='Same', 
                       input_shape=[img_rows, img_cols, num_channel]))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(128, [7, 7], padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(192, [3, 3], padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Convolution2D(256, [3, 3], padding='Same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(4096, kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(4096, kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(breed_count.shape[0], kernel_initializer='normal'))
model.add(BatchNormalization())
model.add(Activation('softmax'))


# In[15]:


model.compile(optimizer='adam', loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(X_train, y_train,
          batch_size=20,
          validation_data=[X_val, y_val],
          epochs=epochs,
          verbose=2)


# In[ ]:




