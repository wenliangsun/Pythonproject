"""
数据读取
TensorFlow程序读取数据一共有3种方法:
    供给数据(Feeding)： 在TensorFlow程序运行的每一步， 让Python代码来供给数据。
    从文件读取数据： 在TensorFlow图的起始， 让一个输入管线从文件中读取数据。
    预加载数据： 在TensorFlow图中定义常量或变量来保存所有数据(仅适用于数据量比较小的情况)。
"""
import tensorflow as tf

