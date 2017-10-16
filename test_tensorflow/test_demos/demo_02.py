import tensorflow as tf
"""
# 创建一个1×2的矩阵
matrix1 = tf.constant([[3., 3.]])
# 创建一个2×1的矩阵
matrix2 = tf.constant([[2.], [2.]])
# 相乘
product = tf.matmul(matrix1, matrix2)

# 启动默认图
# sess = tf.Session()
# res = sess.run(product)
# print(res)
# sess.close()
with tf.Session() as sess:
    # 对product加括号时，会显示详细信息 [array([[ 12.]], dtype=float32)]
    res = sess.run([product])
    print(res)
"""

# 测试如何调用其他的gpu来进行计算
with tf.Session() as sess:
    with tf.device("/gpu:0"): # 启动第一个gpu进行计算
        matrix1 = tf.constant([[3.,3.]])
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.matmul(matrix1,matrix2)
        print(sess.run([product]))
