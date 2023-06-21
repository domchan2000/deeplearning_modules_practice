import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, Flatten, GlobalAvgPool2D
from keras import Model

#functional approach:
def functional_model():

        my_input = Input(shape=(28,28,1))
        # 28x28 image with one channel (grayscale)
        x = Conv2D(32, (3,3), activation='relu')(my_input)
        x = Conv2D(64, (3,3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(128, (3,3), activation='relu')(x)
        x = MaxPool2D()(x)
        x = BatchNormalization()(x)
        
        x = GlobalAvgPool2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dense(10, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=my_input, outputs=x)
        return model

#tf.keras.model: inherit from this class    
class MyCustomModel(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = Conv2D(32, (3,3), activation='relu')
        self.conv2 = Conv2D(64, (3,3), activation='relu')
        self.maxpool1 = MaxPool2D()
        self.batchnorm1 = BatchNormalization()
        
        self.conv3 = Conv2D(128, (3,3), activation='relu')
        self.maxpool2 = MaxPool2D()
        self.batchnorm2 = BatchNormalization()
        
        self.globalavgpool1 = GlobalAvgPool2D()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(10, activation='softmax')
        
    def call(self,my_input):
        x = self.conv1(my_input)
        x = self.conv2(x)
        x = self.maxpool1(x)
        x = self.batchnorm1(x)
        
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.batchnorm2(x)
        
        x = self.globalavgpool1(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x
    
    
def streetsigns_model(nbr_classes):
    #define the input shape of our images and their color channels
    my_input = Input((60, 60 , 3))
    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    # x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)
    
    return Model(inputs = my_input, outputs=x)


# def create_generators(batch)

if __name__ == "__main__":
    model = streetsigns_model(10)
    model.summary()