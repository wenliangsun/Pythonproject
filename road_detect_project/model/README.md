#Road Detect
## introduce
###install 
```bash
$ sudo pip3 install tensorflow-gpu
```
### python test demo
```python
import tensorflow as tf
a = tf.constant([3])
b = tf.constant([6])
c = a+b
with tf.Session() as sess:
    res = sess.run(c)
    print(res)
```
