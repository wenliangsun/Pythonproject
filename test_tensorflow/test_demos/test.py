import tensorflow as tf

print('This is a test demo about tensorflow-gpu')
a = tf.constant(2)
b = tf.constant(3)
# sess = tf.Session()
# print(sess.run(a + b))
tmp = tf.Variable(tf.zeros([1]))
tmp = tmp.assign([9])
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(tmp))