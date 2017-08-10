import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import utils


# Model Network
keepProb = tf.placeholder(tf.float32)

# Supervised learning Image
xImages = tf.placeholder(tf.float32, [None, 2048], name='XImages_placeholder')
w0_1 = tf.Variable(tf.random_normal([2048, 1000]))
b0_1 = tf.Variable(tf.random_normal([1000]))
o0_1 = tf.nn.tanh(tf.add(tf.matmul(xImages, w0_1), b0_1))
w1_1 = tf.Variable(tf.random_normal([1000, 500]))
b1_1 = tf.Variable(tf.random_normal([500]))
o1_1 = tf.nn.tanh(tf.add(tf.matmul(o0_1, w1_1), b1_1))
w2_1 = tf.Variable(tf.random_normal([500, 100]))
b2_1 = tf.Variable(tf.random_normal([100]))
o2_1 = tf.nn.tanh(tf.add(tf.matmul(o1_1, w2_1), b2_1))

# Transform function
w3_1 = tf.Variable(tf.random_normal([100, 50]))
b3_1 = tf.Variable(tf.random_normal([50]))
output = tf.add(tf.matmul(o2_1, w3_1), b3_1)
output = tf.nn.dropout(output, keepProb)
output = tf.nn.l2_normalize(output, 1, epsilon=1e-12)

# Supervised learning Text
xTexts = tf.placeholder(tf.float32, [None, 300], name='XTexts_placeholder')
w1_2 = tf.Variable(tf.random_normal([300, 100]))
b1_2 = tf.Variable(tf.random_normal([100]))
o1_2 = tf.nn.tanh(tf.add(tf.matmul(xTexts, w1_2), b1_2))

# Transform function
w2_2 = tf.Variable(tf.random_normal([100, 50]))
b2_2 = tf.Variable(tf.random_normal([50]))
output1 = tf.add(tf.matmul(o1_2, w2_2), b2_2)
output1 = tf.nn.dropout(output1, keepProb)
output1 = tf.nn.l2_normalize(output1, 1, epsilon=1e-12)


# xImages = tf.placeholder(tf.float32, [None, 2048], name='XImages_placeholder')
# w1_1 = tf.Variable(tf.random_normal([2048, 200]))
# b1_1 = tf.Variable(tf.random_normal([200]))
# output = tf.add(tf.matmul(xImages, w1_1), b1_1)
# output = tf.nn.dropout(output, keepProb)
# output = tf.nn.l2_normalize(output, 1, epsilon=1e-12)
#
# xTexts = tf.placeholder(tf.float32, [None, 300], name='XTexts_placeholder')
# w1_2 = tf.Variable(tf.random_normal([300, 200]))
# b1_2 = tf.Variable(tf.random_normal([200]))
# output1 = tf.add(tf.matmul(xTexts, w1_2), b1_2)
# output1 = tf.nn.dropout(output1, keepProb)
# output1 = tf.nn.l2_normalize(output1, 1, epsilon=1e-12)


# Supervised Loss function
yLabels = tf.placeholder(tf.int32, [None], name='YLabels_placeholder')
oneHotClass = tf.placeholder(tf.int32)
dotProduct = tf.matmul(output, tf.transpose(output1))
entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dotProduct, labels=tf.one_hot(yLabels, oneHotClass)))
# entropy = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(dotProduct, tf.one_hot(yLabels, oneHotClass)), 1))


# Auto encoder Image
deW1_1 = tf.Variable(tf.random_normal([100, 500]))
deB1_1 = tf.Variable(tf.random_normal([500]))
deO1_1 = tf.nn.tanh(tf.add(tf.matmul(o2_1, deW1_1), deB1_1))
deW2_1 = tf.Variable(tf.random_normal([500, 2048]))
deB2_1 = tf.Variable(tf.random_normal([2048]))
deO2_1 = tf.nn.tanh(tf.add(tf.matmul(deO1_1, deW2_1), deB2_1))

# Auto encoder Text
deW1_2 = tf.Variable(tf.random_normal([100, 300]))
deB1_2 = tf.Variable(tf.random_normal([300]))
deO1_2 = tf.nn.tanh(tf.add(tf.matmul(o1_2, deW1_2), deB1_2))

# Construction Loss
construction = tf.reduce_mean(tf.square(xImages - deO2_1)) + tf.reduce_mean(tf.square(xTexts - deO1_2))


# Cross Modality Distributions Matching
kernel = utils.gaussian_Hkernel_matrix
modality = tf.reduce_mean(kernel(o2_1, o2_1))
modality += tf.reduce_mean(kernel(o1_2, o1_2))
modality -= 2 * tf.reduce_mean(kernel(o2_1, o1_2))
modality = tf.where(modality > 0, modality, 0)


# Optimizer
combine = entropy + 1.0*(construction + 0.1*modality)
optimizer = tf.train.AdamOptimizer(1e-3).minimize(combine)


# Predict output
predict = tf.argmax(dotProduct, 1)


# Load data
trainX = np.load(open("trainX100.pkl", "rb"))
trainY = np.load(open("trainY100.pkl", "rb"))
trainZ = np.load(open("trainZ100.pkl", "rb"))
testX = np.load(open("testX100.pkl", "rb"))
testY = np.load(open("testY100.pkl", "rb"))
testZ = np.load(open("testZ100.pkl", "rb"))


# Run Model
np.set_printoptions(threshold=np.nan, suppress=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
miniBatchSize = 1000
for loop in range(10000):
    losses = []
    for i in range(0, trainX.shape[0], miniBatchSize):
        xBatch = trainX[i:i + miniBatchSize]
        yBatch = trainY[i:i + miniBatchSize]
        tmp, _, loss = sess.run([dotProduct, optimizer, entropy], feed_dict={xImages:xBatch, yLabels:yBatch, xTexts:trainZ, oneHotClass:94, keepProb:0.7})
        losses.append(loss)

    totalLoss = sum(losses) / len(losses)
    predictTrain = sess.run(predict, feed_dict={xImages:trainX, xTexts:trainZ, keepProb:1.0})
    tmpTest, predictTest = sess.run([dotProduct, predict], feed_dict={xImages:testX, xTexts:testZ, keepProb:1.0})

    trainAcc = np.mean(np.equal(predictTrain, trainY))
    testAcc = np.mean(np.equal(predictTest, testY))
    print "{0:.5f}, {1:.5f}, {2:.5f}".format(trainAcc, testAcc, totalLoss)