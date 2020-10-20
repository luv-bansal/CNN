import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier=Sequential()

classifier.add(Convolution2D(64,3,3,input_shape=(128,128,3),activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Convolution2D(64,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim=256,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255 ,zoom_range=0.2,horizontal_flip=True,shear_range=0.2)
test_datagen=ImageDataGenerator(rescale=1/255)
training_set=train_datagen.flow_from_directory('training_set',target_size=(128,128),batch_size=32,class_mode='binary')
test_set=train_datagen.flow_from_directory('test_set',target_size=(128,128),batch_size=32,class_mode='binary')

classifier.fit_generator(training_set,samples_per_epoch=8000,nb_epoch=25,validation_data=test_set,nb_val_samples=2000)