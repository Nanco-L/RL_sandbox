import numpy as np
import tensorflow as tf

class FCN(tf.keras.Model):
    def __init__(self, io, hidden):
        super(FCN, self).__init__()
        
        self.io = io
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
    def train_step(self, state, action, value):
        with tf.GradientTape() as tape:
            # TODO: 
            predictions = self.model(state)
            predictions *= tf.one_hot(action, self.model.io)
            
            #print(y)
            #print(predictions)
            loss = self.loss_object(value, tf.reduce_sum(predictions, axis=1))
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)

    def run(self, epochs, train_ds):
        for epoch in range(epochs):
            for state, action, value in train_ds:
                self.train_step(state, action, value)

            if epoch % 10 == 0:
                print(f'{self.train_loss.result()}')

