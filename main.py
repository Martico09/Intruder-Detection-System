#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
import logging
logging.getLogger('tensorflow').disabled = True

from keras.layers import Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from keras import layers
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg16 import VGG16
import cv2

from tqdm import tqdm
import os
import gc
from playsound import playsound


# In[2]:


images = []
labels = []

main_directory = 'data2'

for animal in tqdm(os.listdir(main_directory)):
    for i in range(len(os.listdir(main_directory + '/' + animal))):
            img = cv2.imread(main_directory + '/' + animal + '/' + os.listdir(main_directory + '/' + animal)[i])
            resized_img = cv2.resize(img,(224,224))
            resized_img = resized_img / 255
            images.append(resized_img)
            labels.append(animal)

images = np.array(images,dtype = 'float32')


# In[3]:


le = preprocessing.LabelEncoder()
le.fit(labels)
class_names = le.classes_
labels = le.transform(labels)

labels = np.array(labels, dtype = 'uint8')
labels = np.resize(labels, (len(labels),1))


# In[4]:


x_train_images, x_test_images, y_train_labels, y_test_labels = train_test_split(images, labels,
                                                                                test_size=0.25, 
                                                                                stratify = labels,
                                                                               random_state=9,
                                                                                shuffle=True)


# In[5]:


from tensorflow.keras.utils import to_categorical
y_cat_train=to_categorical(y_train_labels,2)
y_cat_test=to_categorical(y_test_labels,2)
classes_info = {}
classes = sorted(os.listdir(main_directory))
for name in classes:
    classes_info[name] = len(os.listdir(main_directory + f'/{name}'))
print(classes_info)


# In[7]:


from keras.models import Sequential
from tensorflow.keras import layers
model = Sequential([#layers.Dropout(0.2),
    layers.Conv2D(6, kernel_size=5, strides=1,  activation='tanh', input_shape=x_train_images[0].shape, padding='same'),
    layers.AveragePooling2D(),
    layers.Conv2D(16, kernel_size=5, strides=1, activation='relu', padding='valid'),
    layers.AveragePooling2D(),
    #layers.Conv2D(120, kernel_size=5, strides=1, activation='relu', padding='valid'),
    #layers.AveragePooling2D(),
    layers.Conv2D(120, kernel_size=5, strides=1, activation='relu', padding='valid'),
    layers.AveragePooling2D(),
    layers.Conv2D(16, kernel_size=5, strides=1, activation='relu', padding='valid'),
    layers.Flatten(), 
    layers.Dense(84, activation='relu'),
    #layers.Dense(168, activation='relu'),
    #layers.Dense(84, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"] )
model.summary()


# In[11]:


model.fit(x_train_images,y_train_labels, epochs=5, verbose=2)


# In[108]:


#model.save('1Final Model')


# In[2]:


model=tf.keras.models.load_model('Final Model')


# In[12]:


model.evaluate(x_test_images,y_test_labels)


# In[7]:


q = 80
import matplotlib.pyplot as plt
plt.imshow(x_test_images[q] , cmap = "gray")
result = model.predict(x_test_images)
if result[q]>=0.5:
    print('intruder')
else:
    print('allowed')


# In[16]:


##realtime one
t=1
import cv2
face_classifier=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Note the change
video_capture = cv2.VideoCapture(0)
timer = 0
unknown = 0
wait_time = 100
while True:
    # Capture frame-by-frame
    ret, frames = video_capture.read()
    
    
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        #minSize=(30, 30),
        #flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        timer = 0
        unknown = 0
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frames, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (224, 224))
        color_channeled_image = cv2.cvtColor(face_resize, cv2.COLOR_GRAY2BGR)
        array=np.array(color_channeled_image)
        array=array/255
        img = array.reshape(1,224,224,3)
        result=model.predict(img)
        name='name'
        for j in range(100):
            for i in range(result.shape[1]):
                if result[0][i]<0.75:
                    name='Known'    
                else:
                    name='Unknown'
    
        timer +=1
        if name == 'Unknown':
            unknown +=1
        t=1

        value = str(((unknown)/wait_time)*100)+" %"    
                
        cv2.putText(img=frames, text=value, org=(50,50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=1)            
        cv2.putText(img=frames, text=name, org=(x, y-20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),thickness=1)
        print(result)
        print(f"timer = {timer},unknown = {unknown}")
        if timer>wait_time:
            timer = 0
            if unknown>=wait_time//2:
                playsound('beep.wav')
                t=0
    # Display the resulting frame
    cv2.imshow('Video', frames)

    if (cv2.waitKey(1) & 0xFF == ord('q')) or t==0:
        break
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:




