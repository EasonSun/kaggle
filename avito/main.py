from keras.models import Sequential
from keras.engine.training import Model
from keras.layers import Dense, Dropout, Concatenate, Flatten
from keras.regularizers import l2
from keras.losses import binary_crossentropy
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications import xception, inception_v3
from keras.applications import inception_v3

import numpy as np

import os, gc

def make_final_model(pretrained_model, vec_size, do=0.5, l2_strength=1e-5):
    """
    pretrained_model : The VGG conv only part with loaded weights
    vec_size : The size of the vectorized text vector coming from the bag of words 
    n_classes : How many classes are you trying to classify ? 
    do : 0.5 Dropout probability
    l2_strenght : The L2 regularization strength
    
    output : The full pretrained_model that takes images of size (224, 224) and an additional vector
    of size vec_size as input
    """
    
    ### top_aux_model takes the vectorized text as input
    top_aux_model = Sequential()
    top_aux_model.add(Dense(vec_size, input_shape=(vec_size,), name='aux_input'))

    ### top_model takes output from VGG conv and then adds 2 hidden layers
    top_model = Sequential()
    top_model.add(Flatten(input_shape=pretrained_model.output_shape[1:], name='top_flatter'))
    top_model.add(Dense(256, activation='relu', name='top_relu', W_regularizer=l2(l2_strength)))
    top_model.add(Dropout(do))
    top_model.add(Dense(256, activation='sigmoid', name='top_sigmoid', W_regularizer=l2(l2_strength)))

    ### this is than added to the VGG conv-pretrained_model
    pretrained_model.add(top_model)
    
    ### here we merge 'pretrained_model' that creates features from images with 'top_aux_model'
    ### that are the bag of words features extracted from the text. 
    merged = Concatenate([pretrained_model, top_aux_model])

    ### final_model takes the combined feature vectors and add a sofmax classifier to it
    final_model = Sequential()
    final_model.add(merged)
    final_model.add(Dropout(do))
    final_model.add(Dense(1, activation='sigmoid'))

    return final_model

tmp = VGG16(weights='/Users/Account/Drink/kaggle/avito/pretrained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', 
                        include_top=False, 
                        input_shape=(224, 224, 3))
pretrained_model = Sequential()
for layer in tmp.layers:
    pretrained_model.add(layer)
del tmp
gc.collect()

model = make_final_model(pretrained_model, 2500)
model.compile(optimizer = 'adam',
              loss= binary_crossentropy,
              metrics=['accuracy'])