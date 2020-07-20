from tensorflow import keras 
import tensorflow as tf 


class Neuron(keras.layers.Layer):

    def __init__(self, units = 32, input_dim = 32):

        super(Neuron, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value= w_init(shape = (input_dim, units), dtype = "float32"),
            trainable= True
        )

        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value= b_init(
                shape = (units,), 
                dtype="float32"),
            trainable = True
        )

    def __call__(self, inputs):

        return tf.matmul(inputs, self.w) + self.b




# Test class

x = tf.ones((2,2))
print('\n INPUT : ', x.numpy())
n_output = Neuron(4,2)
y = n_output(x)
print(f"\n OUTPUT \n {y}")