from tensorflow import keras 
import tensorflow as tf 


'''
This is an exmaple of a non trainable layer. 
This layer just adds up the tensor values it gets.

'''


class Sumup(keras.layers.Layer):
    def __init__(self, input_dim):
        super(Sumup, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable =False)

    def __call__(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis = 0))
        return self.total


x = tf.ones((2,2))



sum_v = Sumup(2)
y = sum_v(x)


print("OUTPUT")
print(y.numpy())
