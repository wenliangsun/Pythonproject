"""
变量:创建、初始化 保存和加载 
tf.Variable类  和 tf.train.Saver类
"""
import tensorflow as tf

# 创建
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name='weights')
bias = tf.Variable(tf.zeros([200]), name="biases")

# 初始化
"""变量的初始化必须在模型的其他操作运行之前先明确的完成"""
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

# 还可以由另一个变量初始化
# 你可以直接把已初始化的值作为新变量的初始值，
# 或者把它当做tensor计算得到一个值赋予新变量。
weights_2 = tf.Variable(weights.initial_value(), name='weights_2')
w_twice = tf.Variable(weights.initial_value() * 2, name='w_twice')

# 保存和加载
# 用tf.train.Saver()创建一个Saver来管理模型中的所有变量
v1 = tf.Variable(tf.constant(3), name='v1')
v2 = tf.Variable(tf.constant(5), name='v2')

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
"""可以通过给tf.train.Saver()构造函数传入Python字典，
很容易地定义需要保持的变量及对应名称：键对应使用的名称，
值对应被管理的变量。"""
with tf.Session() as sess:
    sess.run(init_op)

    save_path = saver.save(sess, r"./model.ckpt")

    saver.restore(sess, r"./model.ckpt")  # 恢复模型
