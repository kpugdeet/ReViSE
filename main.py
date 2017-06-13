import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import utils

# Load data
trainXImages = np.load(open("trainXImage.pkl", "rb"))
trainXTexts = np.load(open("trainXText.pkl", "rb"))
trainY = np.load(open("trainY.pkl", "rb"))

print (trainXImages.shape, trainXTexts.shape, trainY.shape)

# For ImageFeature
xImages = tf.placeholder(tf.float32, [None, 2048], name='XImages_placeholder')
# Encode
enW1 = tf.Variable(tf.random_normal([2048, 500]))
enB1 = tf.Variable(tf.random_normal([500]))
enW2 = tf.Variable(tf.random_normal([500, 100]))
enB2 = tf.Variable(tf.random_normal([100]))
layer1 = tf.nn.sigmoid(tf.add(tf.matmul(xImages, enW1), enB1))
layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, enW2),enB2))
# Decoder
deW1 = tf.Variable(tf.random_normal([100, 500]))
deB1 = tf.Variable(tf.random_normal([500]))
deW2 = tf.Variable(tf.random_normal([500, 2048]))
deB2 = tf.Variable(tf.random_normal([2048]))
layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2, deW1), deB1))
layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3, deW2), deB2))
# Loss with Jacobian
lossV = tf.reduce_mean(tf.pow(xImages - layer4, 2))


# For ImageFeature
xTexts = tf.placeholder(tf.float32, [None, 300], name='XTexts_placeholder')
# Encode
enW1 = tf.Variable(tf.random_normal([300, 100]))
enB1 = tf.Variable(tf.random_normal([100]))
layer5 = tf.nn.sigmoid(tf.add(tf.matmul(xTexts, enW1), enB1))
# Decoder
deW1 = tf.Variable(tf.random_normal([100, 300]))
deB1 = tf.Variable(tf.random_normal([300]))
layer6 = tf.nn.sigmoid(tf.add(tf.matmul(layer5, deW1), deB1))
# Loss
lossT = tf.reduce_mean(tf.pow(xTexts - layer6, 2))


# Cross Modality Distributions Matching
kernel = utils.gaussian_Hkernel_matrix
LMmd = tf.reduce_mean(kernel(layer2, layer2))
LMmd += tf.reduce_mean(kernel(layer5, layer5))
LMmd -= 2 * tf.reduce_mean(kernel(layer2, layer5))
LMmd = tf.where(LMmd > 0, LMmd, 0)


# Combine all loss
LReconstruct = lossV + lossT
LUnsupervised = LReconstruct + 0.01 * LMmd
LTotal = 1.0 * LUnsupervised

# Optimizer
optimizer = tf.train.AdamOptimizer(0.01).minimize(LTotal)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for loop in range(10000):
    totalLoss, _ = sess.run([LTotal, optimizer], feed_dict={xImages:trainXImages[:1000], xTexts:trainXTexts})
    print (loop, totalLoss)