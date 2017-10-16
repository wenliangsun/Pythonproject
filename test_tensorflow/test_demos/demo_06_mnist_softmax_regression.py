"""
softmax regression 实现mnist数据集的识别
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r'../MNIST_data', one_hot=True)
# print(mnist.train.images.shape)
# print(mnist.train.images[0])

x = tf.placeholder("float32", [None, 784])

w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, w) + b)

y_ = tf.placeholder("float32", [None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, 'float'))

saver = tf.train.Saver()

init = tf.global_variables_initializer()
"""
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    saver.save(sess, r'./softmax_mnist_model/model.ckpt')
    res = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print(res)
"""
with tf.Session() as sess:
    saver.restore(sess, r'./softmax_mnist_model/model.ckpt')
    res = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print(res)
