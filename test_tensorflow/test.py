import tensorflow as tf

print('This is a test demo about tensorflow-gpu')
a = tf.constant(2)
b = tf.constant(3)
sess = tf.Session()
print(sess.run(a + b))
