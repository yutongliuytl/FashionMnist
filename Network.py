import tensorflow as tf 
import numpy as np 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

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

# Converting the output to binary vector form
def vector(data, output):
    vectors = []
    for element in data:
        x = np.array([0] * output)
        x[element] += 1
        vectors.append(x) 
    return vectors 

# Loading the dataset & allocating the train and test data
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.fashion_mnist.load_data()

label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

# Reshaping the array to 4-dims
train_x = train_x.reshape(-1, 28, 28, 1)
test_x = test_x.reshape(-1, 28, 28, 1)
input_shape = (28, 28, 1)

# Making sure that the values are floats
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

# Normalizing the RGB codes
train_x /= 255
test_x /= 255

# Converting labels to binary vector form
train_y = vector(train_y, 10)
test_y = vector(test_y, 10)

# Setting up the graph
features = tf.placeholder(tf.float32, [None, 28,28,1])
labels = tf.placeholder(tf.float32, [None, 10])

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

prediction = conv_network(features, weights, biases)

# Loss and optimizer functions
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

# Predictions and accuracy calculations
values = tf.equal(tf.argmax(prediction, 1),tf.argmax(labels ,1))
accuracy = tf.reduce_mean(tf.cast(values, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

#Training the data
session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

# Training parameters
iterations = 300
batch_size = 512

for i in range(iterations):

    #Diversifying the data each iteration
    batch_x, batch_y = [], []
    for i in range(batch_size):
        i = random.randint(0, train_x.shape[0]-1)
        batch_x.append(train_x[i])
        batch_y.append(train_y[i])

    _, l, a = session.run([optimizer, loss, accuracy], feed_dict={features: batch_x, labels: batch_y})
    print("Loss: " + "{:.6f}".format(l), " Accuracy: " + "{:.5f}".format(a))

# Testing the accuracy of the trained neural network
a = session.run(accuracy, feed_dict={features: test_x, labels: test_y})
print("Test accuracy: " + str(a))

# Save the variables to disk
save_path = saver.save(session, "/tmp/model.ckpt")
print("Model saved in path: %s" % save_path)

session.close()
