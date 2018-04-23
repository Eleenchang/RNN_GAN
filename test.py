import tensorflow as tf

#a = tf.Variable('a')
#b = tf.Variable('b')
a = tf.placeholder(dtype=tf.int64)
b = tf.placeholder(dtype=tf.int64)
c = a + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(c, feed_dict= {a : 1, b : 2}))