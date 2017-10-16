"""
测试变量
"""

import tensorflow as tf

# 创建一个变量，初始化为0
state = tf.Variable(0, name="counter")

# 创建一个op，其作用是使state加1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()
# 启动图之后必须先初始化
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(10):
        sess.run(update)
        print(sess.run(state))


