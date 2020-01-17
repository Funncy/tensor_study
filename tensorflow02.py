import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

# x_train = [1, 2, 3]
# y_train = [1, 2, 3]
X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Our hypothesis XW+b
# hypothesis = x_train * W + b
hypothesis = X * W + b

# cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - y_train))
cost = tf.reduce_mean(tf.square(hypothesis - Y))

#Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    # Fit the Line with new training data
    for step in range(5000):
        cost_val, W_val, b_val, _ = \
            sess.run([cost, W, b, train], feed_dict={X: [1, 2, 3, 4, 5, 6, 7], Y: [2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1]})
        # sess.run(train)
        if step % 20 == 0:
            # print(step, sess.run(cost), sess.run(W), sess.run(b))
            print(step, cost_val, W_val, b_val)
    
    # Testing our model
    print(sess.run(hypothesis, feed_dict={X:[5]}))
    print(sess.run(hypothesis, feed_dict={X:[1.5, 3.6]}))