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
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    @tf.function
    def train_step(self, s, a, r, s_):
        with tf.GradientTape() as tape:
            # TODO: 
            predictions = self.model(s)
            
            #print(y)
            #print(predictions)
            loss = self.loss_object(y, predictions)
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)

    def run(self, epochs, train_ds):
        for epoch in range(epochs):
            for s, a, r, s_ in train_ds:
                self.train_step(s, a, r, s_)

            if epoch % 10:
                print(f'{self.train_loss.result()}')

