# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import tensorflow as tf
import tensorflowvisu
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

# Convo


# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])

stddev=0.1

n1 = 200
# weights W[784, n1]   784=28*28
#W1 = tf.Variable(tf.truncated_normal([784, n1], stddev=stddev))
#b1 = tf.Variable(tf.ones([n1])/10)

# three convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 10 softmax neurons)
K = 4  # first convolutional layer output depth
L = 8  # second convolutional layer output depth
M = 12  # third convolutional layer
N = 200  # fully connected layer


W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))  # 5x5 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.ones([K])/10)

n2 = 100
W2 = tf.Variable(tf.truncated_normal([3136, n2], stddev=stddev))
B2 = tf.Variable(tf.ones([n2])/10)

n3 = 60
W3 = tf.Variable(tf.truncated_normal([n2, n3], stddev=stddev))
B3 = tf.Variable(tf.ones([n3])/10)

n4 = 30
W4 = tf.Variable(tf.truncated_normal([n3, n4], stddev=stddev))
B4 = tf.Variable(tf.ones([n4])/10)

out = 10
# weights W[784, 10]   784=28*28
W = tf.Variable(tf.truncated_normal([n4, out], stddev=stddev))
# biases b[10]
B = tf.Variable(tf.ones([out])/10)

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 784])

# The model

stride = 1  # output is 28x28
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)

# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y1, shape=[-1, 28 * 28 * K])

#Y1 = tf.nn.relu(tf.matmul(XX, W1) + b1)
Y2 = tf.nn.relu(tf.matmul(YY, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W) + B
Y = tf.nn.softmax(Ylogits)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Ylogits)
cross_entropy = tf.reduce_mean(cross_entropy) * 100.0  # normalized for batches of 100 images,

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# decaying learning rate
lr = tf.placeholder(tf.float32)
# training
train_step = tf.train.MomentumOptimizer(lr, 0.8).minimize(cross_entropy)

# matplotlib visualisation
allweights = tf.reshape(W, [-1])
allbiases = tf.reshape(B, [-1])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)


    # compute training values for visualisation
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate})



datavis.animate(training_step, iterations=10000+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(2000+1): training_step(i, i % 50 == 0, i % 10 == 0)

print("max test accuracy: " + str(datavis.get_max_test_accuracy()))

# final max test accuracy = 0.9268 (10K iterations). Accuracy should peak above 0.92 in the first 2000 iterations.
