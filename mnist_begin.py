import tensorflow as tf 
from tensorflow import keras 


class MNIST(keras.models.Model):

    def __init__(self):

        super(MNIST, self).__init__()
        self.layer1 = keras.layers.Dense(784)
        self.layer2 = keras.layers.Dense(64)
        self.layer3 = keras.layers.Dense(10)

    def __call__(self, inputs):

        x = keras.layers.Flatten()(inputs)
        x = self.layer1(x)
        x = tf.nn.relu(x)
        x = self.layer2(x)
        x = tf.nn.relu(x)

        return tf.nn.softmax(self.layer3(x))




mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, x_test = x_train/255.0, x_test/255.0
train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(32)


model = MNIST()

loss_f = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name = 'train_loss')


@tf.function
def train_step(images, labels):

    with tf.GradientTape() as tape:

        predictions = model(images)
        loss = loss_f(labels, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)

EPOCHS = 5

for epoch in range(EPOCHS):
    
    train_loss.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)

    print(f" Epoch: {epoch +1 } \n Loss: {train_loss.result()} ")