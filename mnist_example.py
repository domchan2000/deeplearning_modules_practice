import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D

from deeplearning_models import MyCustomModel, functional_model
from my_utils import display

# tensorflow.keras.Sequential
seq_model = tf.keras.Sequential(
    [
        Input(shape=(28,28,1)),
        # 28x28 image with one channel (grayscale)
        Conv2D(32, (3,3), activation='relu'),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPool2D(),
        BatchNormalization(),
        
        GlobalAvgPool2D(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ]
)


if __name__ == '__main__':
    
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("x_train.shape = ", x_train.shape)
    print("x_train.shape = ", y_train.shape)
    print("x_train.shape = ", x_test.shape)
    print("x_train.shape = ", y_test.shape)
    
    if False:
        display(x_train,y_train)
    
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    #if using one-hot encodeing
    y_train = tf.keras.utils.to_categorical(y_train,10)
    y_test = tf.keras.utils.to_categorical(y_test,10)    
    
    # model = functional_model()
    model = MyCustomModel()
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')
    

    #model training
    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2)
    
    #evaluation on test set
    model.evaluate(x_test, y_test, batch_size=64)
    
