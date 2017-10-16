"""
测试feed
feed 使用一个 tensor 值临时替换一个操作的输出结果. 
你可以提供 feed 数据作为 run() 调用的参数.
feed 只在调用它的方法内有效, 方法结束, feed 就会消失. 
最常见的用例是将某些特殊的操作指定为 "feed" 操作, 
标记的方法是使用 tf.placeholder() 为这些操作创建占位符. 
"""
import tensorflow as tf

input1 = tf.placeholder("float32")
input2 = tf.placeholder("float32")
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    print(sess.run([output],
                   feed_dict={input1:[7.],input2:[2.]}))