# -*- coding: utf-8 -*-

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import load_model
from keras.layers import Dense, GlobalAveragePooling2D
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

NUM_SAMPLE = 1200
BATCH_SIZE =30
#this is to print pred_class in correct format by suppressing e-1...
np.set_printoptions(precision=3, suppress=True)

ModelPath = r'C:\Users\Karim El Guermai\Desktop\PROGRAMMING\FinalProject\Karim\CNN_model.h5'
imagePath = r'C:\Users\Karim El Guermai\Desktop\PROGRAMMING\FinalProject\Karim\Gestures_splitten'
#%%
# create_img_generator is for data augmentation, it is optional
def create_img_generator():
    return ImageDataGenerator()    
    
def generateData(path):
    #If you're not using data augmentation, ImageDataGenerator().flow_from_direcotry()
    predict_generator = create_img_generator().flow_from_directory(
            path,
            target_size = (50, 50),
            batch_size = BATCH_SIZE ,
            color_mode='grayscale',   #it is by default 'rgb'. 1 means grayscale
            #you set classes and class_mode to none if you don't have the label of the samples
           # classes = None,
           # class_mode=None, #if you have image classes mixed
            #save_to_dir # store augmented data to save_to_dir
            #classes = ['karim','A','DEL', if you don't specify it, classes names will be inferred form the directory structure],
            seed = 42)   
    return predict_generator

#%%
model = load_model(ModelPath)
test_batches = generateData(ModelPath)
#test_imgs, labels = next(test_batches) no need as we grab all test_batches at once
predictions = model.predict_generator(test_batches, steps=NUM_SAMPLE/BATCH_SIZE) #STEPS= #of batches
#if you omit interpolation,il will not work
#plt.imshow(test_imgs[4], cmap = 'gray', interpolation = 'bicubic')
#%%
