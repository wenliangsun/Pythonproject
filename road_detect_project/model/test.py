import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from road_detect_project.model.data import AerialDataset
from road_detect_project.model.params import dataset_params


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


dataset = AerialDataset()
path = r"/media/sunwl/sunwl/datum/roadDetect_project/Massachusetts/"
params = dataset_params
dataset.load(path, params=params)
dataset.switch_active_training_set(0)
train_data = dataset.data_set['train']
print("train")
print(train_data[0].shape)
print(train_data[1].shape)
valid_data = dataset.data_set['valid']
print("valid")
print(valid_data[0].shape)
print(valid_data[1].shape)
test_data = dataset.data_set['test_PyQt5']
print("test_PyQt5")
print(test_data[0].shape)
print(test_data[1].shape)
print(dataset.get_report())
dataset.switch_active_training_set(2)
dataset.get_elements(2)
new_data = dataset.data_set['train']
print(new_data[0].shape)

batch_size = 16

keep_prob = tf.placeholder("float32", shape=[3])
x = tf.placeholder("float32", shape=[None, 64, 64, 3])
y_ = tf.placeholder("float32", shape=[None, 16, 16])
w_conv1 = weight_variable([13, 13, 3, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
h_pool1 = max_pool_2d(h_conv1)

w_conv2 = weight_variable([4, 4, 64, 112])
b_conv2 = bias_variable([112])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_drop2 = tf.nn.dropout(h_conv2, keep_prob=keep_prob[0])

w_conv3 = weight_variable([3, 3, 112, 80])
b_conv3 = bias_variable([80])
h_conv3 = tf.nn.relu(conv2d(h_drop2, w_conv3) + b_conv3)
h_drop3 = tf.nn.dropout(h_conv3, keep_prob=keep_prob[1])

w_fc1 = weight_variable([32 * 32 * 80, 512])
b_fc1 = bias_variable([512])
h_drop3_flat = tf.reshape(h_drop3, [-1, 32 * 32 * 80])
h_fc1 = tf.nn.relu(tf.matmul(h_drop3_flat, w_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob[2])

w_fc2 = weight_variable([512, 256])
b_fc2 = bias_variable([256])

y = tf.nn.sigmoid(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
y_out = tf.reshape(y, [-1, 16, 16])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_out + 1e-10))
train_step = tf.train.GradientDescentOptimizer(0.0014).minimize(cross_entropy)
correct_prediction = tf.equal(y_ > 0.5, y_out > 0.5)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    chunks = dataset.get_chunk_number()
    nr_iter = 0
    for i in range(1000):
        for chunk in range(chunks):
            dataset.switch_active_training_set(chunk)
            nr_elements = dataset.get_elements(chunk)
            train_data = dataset.data_set['train']
            batches = [[train_data[0][x:x + batch_size], train_data[1][x:x + batch_size]]
                       for x in range(0, nr_elements, batch_size)]
            for batch in batches:
                nr_iter += 1
                # print(batch[1][1])
                log = sess.run([cross_entropy, train_step],
                               feed_dict={x: batch[0],
                                          y_: batch[1],
                                          keep_prob: [0.9, 0.8, 0.5]})
                print(log[0])
                print("***********************************************")
                if nr_iter % 1 == 0:
                    temp_acc = sess.run(y_out,
                                feed_dict={x: batch[0],
                                           keep_prob: [1.0, 1.0, 1.0]})
            # print("step %d, training accuracy %g" % (i, temp_acc))
        print("one Epoch complete")





# with tf.Session() as sess:
#     sess.run(init)
#     res = sess.run([y_out], feed_dict={x: data})
#     # print(res)
#     # print(res[0].shape)
#     img = res[0] >= 0.5
#     # print(img)
#     img = img * 255
#     img = img.astype('uint8')
#     print(img.dtype)
#     plt.imshow(img[0], cmap='gray')
#     plt.show()
