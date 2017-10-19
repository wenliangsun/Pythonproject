import numpy as np
import tensorflow as tf

a = tf.Variable([0.])
a = a.assign([9])
c = tf.equal(np.random.rand(16, 16) > 0.5, np.random.rand(16, 16) > 0.5)
d = tf.reduce_mean(tf.cast(c,"float32"))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(d))
