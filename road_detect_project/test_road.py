import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data = np.float32(np.random.rand(2, 64, 64, 3))


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


x = tf.placeholder("float32", shape=[None, 64, 64, 3])
# y_ = tf.placeholder("float32", shape=[None, 16, 16])

w_conv1 = weight_variable([13, 13, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
h_pool1 = max_pool_2d(h_conv1)

w_conv2 = weight_variable([4, 4, 64, 112])
b_conv2 = bias_variable([112])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_drop2 = tf.nn.dropout(h_conv2, keep_prob=0.9)

w_conv3 = weight_variable([3, 3, 112, 80])
b_conv3 = bias_variable([80])
h_conv3 = tf.nn.relu(conv2d(h_drop2, w_conv3) + b_conv3)
h_drop3 = tf.nn.dropout(h_conv3, keep_prob=0.8)

w_fc1 = weight_variable([32 * 32 * 80, 512])
b_fc1 = bias_variable([512])
h_drop3_flat = tf.reshape(h_drop3, [-1, 32 * 32 * 80])
h_fc1 = tf.nn.relu(tf.matmul(h_drop3_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=0.5)

w_fc2 = weight_variable([512, 256])
b_fc2 = bias_variable([256])

y = tf.nn.sigmoid(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
y_out = tf.reshape(y, [-1, 16, 16])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    res = sess.run([y_out], feed_dict={x: data})
    # print(res)
    # print(res[0].shape)
    img = res[0] >= 0.5
    # print(img)
    img = img * 255
    img = img.astype('uint8')
    print(img.dtype)
    plt.imshow(img[0], cmap='gray')
    plt.show()
