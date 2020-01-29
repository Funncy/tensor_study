import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

#Logistic classifier

xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

nb_classes = 7

#placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, nb_classes) # one hot , rank + 1
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes]) # rank - 1

W = tf.Variable(tf.random_normal([16, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')


# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
logits = tf.matmul(X, W) + b

# Cross entropy cost/loss
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

prediction = tf.argmax(hypothesis, 1)
correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 

# Train the model
# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10000):
            sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
            if step % 100 == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={X: x_data, Y: y_data})
                print("Step: {:5}\tLostt: {:.3f}\tAcc: {:.2%}".format(step, loss, acc))

    print('--------------')
    # Let's See if we can predict
    pred = sess.run(prediction, feed_dict={X: x_data})

    # y_data: (N, 1) = flatten => (N, ) matches pred.shape
    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Predction: {} True Y: {}".format(p == int(y), p, int(y)))