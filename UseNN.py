import tensorflow as tf
import numpy as np

# Loading the dataset & allocating the train and test data
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

# Convolution wrapper with bias and relu activation
def conv(x, weights, biases, strides=1):
    x = tf.nn.conv2d(x, weights, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, biases)
    return tf.nn.relu(x) 

# Max pooling wrapper
def maxpool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def conv_network(x, weights, biases):  
    # Input layer (ouputs a 14 x 14)
    conv1 = conv(x, weights['wc1'], biases['bc1'])
    conv1 = maxpool(conv1)

    # Convolution Layer (outputs a 7 x 7 and then a 4 x 4)
    conv2 = conv(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool(conv2)
    conv3 = conv(conv2, weights['wc3'], biases['bc3'])
    conv3 = maxpool(conv3)

    # Dense layer
    dl = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    dl = tf.add(tf.matmul(dl, weights['wd1']), biases['bd1'])
    dl = tf.nn.relu(dl)

    # Output layer
    output = tf.add(tf.matmul(dl, weights['out']), biases['out'])
    return output

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128, 10), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}

# Setting up the graph
features = tf.placeholder(tf.float32, [None, 28,28,1])
labels = tf.placeholder(tf.float32, [None, 10])
prediction = conv_network(features, weights, biases)

# Reloading the model
session = tf.Session()
saver = tf.train.Saver()

# Restore variables from disk
saver.restore(sess = session, save_path = "/tmp/model.ckpt")
print("Model restored.")

# Predictions
predict_dataset = test_x[0].reshape(-1, 28, 28, 1)
predict_dataset = predict_dataset.astype('float32') / 255

classification = session.run(tf.argmax(prediction, 1), feed_dict={features: predict_dataset})
print("Prediction: {}".format(classification[0]))
print("Label: " + str(test_y[0]))

