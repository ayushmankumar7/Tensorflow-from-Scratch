from tensorflow import keras 
import tensorflow as tf 

# Calling the Linear Class from common_way.py
from common_way import Linear

class MLPBlock(keras.layers.Layer):

    def __init__(self):
        super(MLPBlock, self).__init__()
        self.linear_1 = Linear(32)
        self.linear_2 = Linear(32)
        self.linear_3 = Linear(1)

    def __call__(self, inputs):
        
        x = self.linear_1(inputs)
        x = tf.nn.relu(x)
        x = self.linear_2(x)
        x =tf.nn.relu(x)
        return self.linear_3(x)

    

mlp = MLPBlock()
y = mlp(tf.ones(shape =(3,64)))

print('Weights' , mlp.weights)
print("trainable parameters", len(mlp.trainable_weights))