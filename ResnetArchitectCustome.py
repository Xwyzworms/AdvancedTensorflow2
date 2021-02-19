# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:27:06 2021

@author: xwyzworm
"""
#%%
import tensorflow as tf
import tensorflow_datasets as tfds
import pydot
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
#%%

class firstIdentityBlock(tf.keras.Model):
    
    def __init__(self,filters : int, kernel_size : int):
        super(firstIdentityBlock,self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters,kernel_size,padding="SAME")
        self.batchNorm1 = tf.keras.layers.BatchNormalization()
                
        self.activation = tf.keras.layers.Activation(tf.nn.relu)
        self.add = tf.keras.layers.Add()
        
        
    def call (self, input_tensors): 
        
        layer = self.conv1(input_tensors)
        layer = self.batchNorm1(layer)
        layer = self.activation(layer)
        
        layer = self.conv1(layer)
        layer = self.batchNorm1(layer)
        layer = self.activation(layer)
        
        layer = self.add([layer,input_tensors])
        layer = self.activation(layer)
        
        return layer
        
#%%        

class secondIdentityBlock(tf.keras.Model):
    
    def __init__(self, filters ,kernel_size):
        super(secondIdentityBlock,self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters,kernel_size)
        self.batchNorm1 = tf.keras.layers.BatchNormalization()
        
        self.activation = tf.keras.layers.Activation(tf.nn.relu)
        self.additionConv = tf.keras.layers.Conv2D(filters,kernel_size,padding="SAME")
        self.addGan = tf.keras.layers.Add()
        
    def call (self, input_tensors):
        
        secLayer = self.additionConv(input_tensors)
        
        layer = self.conv1(input_tensors)
        layer = self.batchNorm1(layer)
        layer = self.activation(layer)
        
        layer = self.conv1(layer)
        layer = self.batchNorm1(layer)
        layer = self.addGan([secLayer,layer])
        
        layer = self.activation(layer)
        
        return layer
class ResNet(tf.keras.Model):
    
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3, 3))

        # Use the Identity blocks that you just defined
        self.id1a = firstIdentityBlock(64, 5)
        self.id1b = secondIdentityBlock(64, 5)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        # insert the identity blocks in the middle of the network
        x = self.id1a(x)
        for i in range(0,2):
            x = self.id1b(x)

        x = self.global_pool(x)
        return self.classifier(x)
    

# utility function to normalize the images and return (image, label) pairs.
def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']

# create a ResNet instance with 10 output units for MNIST
resnet = ResNet(10)
resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# load and preprocess the dataset
dataset = tfds.load('mnist', split=tfds.Split.TRAIN)
dataset = dataset.map(preprocess).batch(32)

# train the model.
history = resnet.fit(dataset, epochs=1)

#%%

plot_model(resnet)

#%%
print(history.history['accuracy'])