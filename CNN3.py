# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 01:47:30 2019

@author: Pranav
"""

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnsit = input_data.read_data_sets("MNSIT_data/", one_hot=True)
num_input = 28*28*1
num_classes = 10

x_ = tf.placeholder("float", shape=[None, num_input], name='X')
y_ = tf.placeholder("float", shape=[None, num_classes], name='Y')

is_training = tf.placeholder(tf.bool)
x_image = tf.reshape(x_, [-1,28,28,1])

#Conv1
conv1 = tf.layers.conv2d(inputs=x_image, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu)

#Pool1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

#Conv2
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)

#Pool2
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

#Flatten
pool_2_flat = tf.reshape(pool2, [-1,7*7*64])

#Dense Layer
dense = tf.layers.dense(inputs=pool_2_flat, units=1024, activation=tf.nn.relu)

#Dropout
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=is_training)

#Logits Layer
logits = tf.layers.dense(inputs=dropout, units=10)

#Loss Function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Build graph
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Train
for i in range(2000):
    batch = mnsit.train.next_batch(50)
    
    if i % 100 == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x_:batch[0], y_:batch[1], is_training: True})
        print("Step %d, Training Accuracy %g "%(i, train_accuracy))
        
    train_step.run(session=sess, feed_dict={x_:batch[0], y_:batch[1], is_training: False})
    
    print("Test Accuracy:",sess.run(accuracy, feed_dict={x_: mnsit.test.images, y_: mnsit.test.labels, is_training: False}))



#98.43-Final Accuracy
