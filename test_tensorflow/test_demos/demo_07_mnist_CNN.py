"""
CNN 实现mnist数据集的识别

一般而言，对于输入张量（input tensor）
有四维信息：[batch, height, width, channels]
（分别表示 batch_size, 也即样本的数目，单个样本的行数和列数，
样本的频道数，rgb图像就是三维的，灰度图像则是一维），
对于一个二维卷积操作而言，其主要作用在 height, width上。

strides参数确定了滑动窗口在各个维度上移动的步数。
一种常用的经典设置就是要求，strides[0]=strides[3]=1。

strides[0] = 1，也即在 batch 维度上的移动为 1，也就是不跳过任何一个样本，否则当初也不该把它们作为输入（input）
strides[3] = 1，也即在 channels 维度上的移动为 1，也就是不跳过任何一个颜色通道；

"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(r'../MNIST_data', one_hot=True)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


x = tf.placeholder("float32", shape=[None, 784])
y_ = tf.placeholder("float32", shape=[None, 10])
# 前两个维度是patch的大小，接着是输入的通道数目，
# 最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2d(h_conv1)

w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2d(h_conv2)

w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

init = tf.global_variables_initializer()

saver = tf.train.Saver()
"""
with tf.Session() as sess:
    sess.run(init)
    for i in range(20000):
        batch = mnist.train.next_batch(30)
        if i % 100 == 0:
            temp_acc = sess.run(accuracy,
                                feed_dict={x: batch[0],
                                           y_: batch[1],
                                           keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, temp_acc))
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    saver.save(sess, r'./mnist_CNN_model/model.ckpt')
"""
test_acc = tf.Variable(tf.zeros([0]))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, r'./mnist_CNN_model/model.ckpt')

    for i in range(1000):
        test_batch = mnist.test.next_batch(30)
        test_acc += sess.run(accuracy, feed_dict={x: test_batch[0],
                                                  y_: test_batch[1],
                                                  keep_prob: 1.0})
        # print("test accuracy %g" % test_acc)
    print("test accuracy is %g" % (float(test_acc[0]) / float(1000)))
