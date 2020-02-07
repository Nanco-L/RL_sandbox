import numpy as np
import tensorflow as tf
import timeit

class FCN(tf.keras.Model):
    def __init__(self, io, hidden, trainable=True):
        super(FCN, self).__init__()
        
        self.io = io
        self.d1 = tf.keras.layers.Dense(hidden, activation='tanh', input_shape=(io,), trainable=trainable)
        self.d2 = tf.keras.layers.Dense(hidden, activation='tanh', trainable=trainable)
        self.d3 = tf.keras.layers.Dense(io, activation='tanh', trainable=trainable)
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
    def __init__(self, t_model, e_model):
        self.t_model = t_model
        self.e_model = e_model
        self.loss_object = tf.keras.losses.MeanSquaredError()
        #self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        #self.optimizer = tf.keras.optimizers.SGD()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.apply_rate = 0.03
    
    @tf.function
    def train_step(self, state, action, value):
        with tf.GradientTape() as tape:
            # TODO: 
            predictions = self.t_model(state)
            predictions *= tf.one_hot(action, self.t_model.io)
            # FIXME: Check the loss -> is it correct?
            loss = self.loss_object(value, tf.reduce_sum(predictions, axis=1))
            #loss = self.loss_object(tf.reshape(value, [-1,1])*tf.one_hot(action, self.t_model.io, dtype=tf.float64), predictions)
            
        gradients = tape.gradient(loss, self.t_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.t_model.trainable_variables))
        self.train_loss(loss)
        #self.apply_networks()

    def run(self, epochs, train_ds, global_epoch, time1=None):
        time1 = timeit.default_timer()
        for epoch in range(epochs):
            for state, action, value in train_ds:
                self.train_step(state, action, value)

            if (global_epoch+epoch+1) % 50 == 0:
                time2 = timeit.default_timer()
                print(f'{epoch+global_epoch+1:6d} th epoch, Train loss: {self.train_loss.result():6.4f}, Elipsed: {time2 - time1:6.4f}')
                time1 = time2
        
        return global_epoch + epoch + 1

    def save(self):
        #self.model.save('my_model', save_model='tf')
        print(self.e_model)
        self.e_model.save_weights('./my_checkpoint')

    def load(self):
        #self.model = tf.keras.load_model('my_model')
        self.train_step(np.random.random([1,9]), np.random.random([1,1]).astype(np.int), np.random.random([1,1]))
        self.e_model.load_weights('./my_checkpoint')
        #print(self.model.d1.get_weights())
        self.e_model.summary()

    def apply_networks(self):
        self.e_model.d1.set_weights([item - self.apply_rate*(item-jtem) for item,jtem in zip(self.e_model.d1.weights, self.t_model.d1.weights)])
        self.e_model.d2.set_weights([item - self.apply_rate*(item-jtem) for item,jtem in zip(self.e_model.d2.weights, self.t_model.d2.weights)])
        self.e_model.d3.set_weights([item - self.apply_rate*(item-jtem) for item,jtem in zip(self.e_model.d3.weights, self.t_model.d3.weights)])

