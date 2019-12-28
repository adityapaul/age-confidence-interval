import tensorflow as tf
import numpy as np
import keras
from keras import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
import os
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

#initial VGG-Face model
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))

#age tracking layers
model.add(Convolution2D(117, (1, 1), name='predictions')) # change to 101 for pretrained weights
model.add(Flatten())
model.add(Activation('softmax'))

#model.load_weights('./weights/age_model_weights.h5')

#get filenames for training set
filenames = os.listdir('./UTKFace')

img_list = []
for i in range(0, len(filenames)):
    rawimg = cv2.resize(cv2.imread('./UTKFace/' + filenames[i]), (224, 224))
    img = cv2.cvtColor(rawimg, cv2.COLOR_BGR2RGB)
    img_list.append(img)
    if i % 5000 == 0:
        print(str(i) + " of " + str(len(filenames)) + " images loaded")

age_list = []
for i in range(0, len(filenames)):
    age_list.append(int(filenames[i].split('_')[0]))
    if i % 5000 == 0:
        print(str(i) + " of " + str(len(filenames)) + " ages loaded")

img_train, img_test, age_train, age_test = train_test_split(img_list, age_list, test_size=0.3)
del img_list
del age_list


print("starting")
img_train = np.asarray(img_train)
print("img_train done")
img_test = np.asarray(img_test)
print("img_test done")
age_train = keras.utils.to_categorical(np.asarray(age_train))
print("age_train done")
age_test = keras.utils.to_categorical(np.asarray(age_test))
print("age_test done")

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(img_train, age_train, batch_size=256, validation_split=0.3, epochs=1, verbose=1)

print("model trained")
