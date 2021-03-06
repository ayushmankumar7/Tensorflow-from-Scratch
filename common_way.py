from tensorflow import keras 
import tensorflow as tf 


class Linear(keras.layers.Layer):
    def __init__(self, units =32):

        super(Linear, self).__init__()
        self.units = units 
    
    def build(self, input_shape):

        self.w = self.add_weight(
            shape= (input_shape[-1], self.units),
            initializer= "random_normal",
            trainable= True
        )

        self.b = self.add_weight(
            shape= (self.units, ),
            initializer= "random_normal",
            trainable= True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b 


x = tf.ones((2,2))
lin = Linear(2)

print(lin(x))