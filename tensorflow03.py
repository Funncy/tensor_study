import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt

tf.disable_v2_behavior()

# X = [1, 2, 3]
# Y = [1, 2, 3]

#W = tf.placeholder(tf.float32)
x_data = [1, 2, 3]
y_data = [1, 2, 3]
W = tf.Variable(tf.random_normal([1]), name='weight')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothsis for linear model X * W
hypothesis = X * W

# cost/loss function
# cost = tf.reduce_mean(tf.square(hypothesis - Y))
cost = tf.reduce_sum(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative : W -= learning_rate * derivative
# 직접 미분 돌려서 결과 출력 
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)
"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(cost)
"""

sess = tf.Session()

sess.run(tf.global_variables_initializer())

# 그래프 찍기
# W_val = []
# cost_val = []

# for i in range(-30, 50):
#     feed_W = i * 0.1
#     curr_cost, curr_W = sess.run([cost, W], feed_dict={W: feed_W})
#     W_val.append(curr_W)
#     cost_val.append(curr_cost)

# plt.plot(W_val, cost_val)
# plt.show()

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

