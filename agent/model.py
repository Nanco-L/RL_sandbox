import numpy as np
import tensorflow as tf

class FCN(tf.keras.Model):
    def __init__(self, io, hidden):
        super(FCN, self).__init__()
        self.d1 = tf.keras.layers.Dense(hidden, activation='sigmoid')
        self.d2 = tf.keras.layers.Dense(hidden, activation='sigmoid')
        self.d3 = tf.keras.layers.Dense(io, activation='linear')
        self.train_loss = tf.keras.metrics.Mean()
        
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
    
class CNN(tf.keras.Model):
    def __init__(self):
        """
        """
        
    def call(self, x):
        return 0

class DQNWrapper():
    def __init__(self, model):
        self.model = model
    
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(x)
            #print(y)
            #print(predictions)
            loss = self.loss_object(y, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.loss_mean(loss)
