import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import utils

# Load data
trainXImages = np.load(open("trainXImage.pkl", "rb"))
trainXTexts = np.load(open("trainXText.pkl", "rb"))
trainY = np.load(open("trainY.pkl", "rb"))
testXImages = np.load(open("testXImage.pkl", "rb"))
testY= np.load(open("testY.pkl", "rb"))
print (trainXImages.shape, trainY.shape, trainXTexts.shape, testXImages.shape, testY.shape,)


# For ImageFeature
xImages = tf.placeholder(tf.float32, [None, 2048], name='XImages_placeholder')
xImagesUnLabel = tf.placeholder(tf.float32, [None, 2048], name='XImagesUnLabel_placeholder')

# Encode
enW1 = tf.Variable(tf.random_normal([2048, 500]))
enB1 = tf.Variable(tf.random_normal([500]))
enW2 = tf.Variable(tf.random_normal([500, 100]))
enB2 = tf.Variable(tf.random_normal([100]))
layer1 = tf.nn.tanh(tf.add(tf.matmul(xImages, enW1), enB1))
layer2 = tf.nn.tanh(tf.add(tf.matmul(layer1, enW2),enB2))
layer1Unlabel = tf.nn.tanh(tf.add(tf.matmul(xImagesUnLabel, enW1), enB1))
layer2Unlabel = tf.nn.tanh(tf.add(tf.matmul(layer1Unlabel, enW2), enB2))

# Decoder
deW1 = tf.Variable(tf.random_normal([100, 500]))
deB1 = tf.Variable(tf.random_normal([500]))
deW2 = tf.Variable(tf.random_normal([500, 2048]))
deB2 = tf.Variable(tf.random_normal([2048]))
layer3 = tf.nn.tanh(tf.add(tf.matmul(layer2, deW1), deB1))
layer4 = tf.nn.tanh(tf.add(tf.matmul(layer3, deW2), deB2))
layer3Unlabel = tf.nn.tanh(tf.add(tf.matmul(layer2Unlabel, deW1), deB1))
layer4Unlabel = tf.nn.tanh(tf.add(tf.matmul(layer3Unlabel, deW2), deB2))

# Loss with Jacobian
lossVLabel = tf.reduce_mean(tf.add(tf.square(xImages - layer4), 0.1 * tf.squeeze(tf.square(tf.gradients(layer2, xImages)))))
lossVUnlabel = tf.reduce_mean(tf.add(tf.square(xImagesUnLabel - layer4Unlabel), 0.1 * tf.squeeze(tf.square(tf.gradients(layer2Unlabel, xImagesUnLabel)))))
lossV = (lossVLabel + lossVUnlabel) / 2.0


# For Text attribute
xTexts = tf.placeholder(tf.float32, [None, 300], name='XTexts_placeholder')

# Encode
enW1Text = tf.Variable(tf.random_normal([300, 100]))
enB1Text = tf.Variable(tf.random_normal([100]))
layer5 = tf.nn.tanh(tf.add(tf.matmul(xTexts, enW1Text), enB1Text))

# Decoder
deW1Text = tf.Variable(tf.random_normal([100, 300]))
deB1Text = tf.Variable(tf.random_normal([300]))
layer6 = tf.nn.tanh(tf.add(tf.matmul(layer5, deW1Text), deB1Text))

# Loss
lossT = tf.reduce_mean(tf.square(xTexts - layer6))


# Cross Modality Distributions Matching
kernel = utils.gaussian_Hkernel_matrix
LMmdLabel = tf.reduce_mean(kernel(layer2, layer2))
LMmdLabel += tf.reduce_mean(kernel(layer5, layer5))
LMmdLabel -= 2 * tf.reduce_mean(kernel(layer2, layer5))
LMmdLabel = tf.where(LMmdLabel > 0, LMmdLabel, 0)
LMmdUnLabel = tf.reduce_mean(kernel(layer2Unlabel, layer2Unlabel))
LMmdUnLabel += tf.reduce_mean(kernel(layer5, layer5))
LMmdUnLabel -= 2 * tf.reduce_mean(kernel(layer2Unlabel, layer5))
LMmdUnLabel = tf.where(LMmdLabel > 0, LMmdLabel, 0)
LMmd = (LMmdLabel + LMmdUnLabel) / 2.0


# Transform function
keepProb = tf.placeholder(tf.float32)
tfW1 = tf.Variable(tf.random_normal([100, 50]))
tfb1 = tf.Variable(tf.random_normal([50]))
tfW2 = tf.Variable(tf.random_normal([100, 50]))
tfb2 = tf.Variable(tf.random_normal([50]))
yImages = tf.nn.dropout(tf.add(tf.matmul(layer2, tfW1), tfb1), keepProb)
yImagesUnlabel = tf.nn.dropout(tf.add(tf.matmul(layer2Unlabel, tfW1), tfb1), keepProb)
yTexts = tf.nn.dropout(tf.add(tf.matmul(layer5, tfW2), tfb2), keepProb)


# L2 normalization on the output scores
yImages = tf.nn.l2_normalize(yImages, 1, epsilon=1e-12)
yImagesUnlabel = tf.nn.l2_normalize(yImagesUnlabel, 1, epsilon=1e-12)
yTexts = tf.nn.l2_normalize(yTexts, 1, epsilon=1e-12)


# For softmax
numClass = tf.placeholder(tf.int32)
numClassUnLabel = tf.placeholder(tf.int32)


# Supervised loss
yLabels = tf.placeholder(tf.int64, [None], name='YLabels_placeholder')
LSupervised = tf.multiply(tf.matmul(yImages,tf.transpose(yTexts[:numClass])), tf.one_hot(yLabels, numClass))
LSupervised = -1 * tf.reduce_mean(tf.reduce_sum(LSupervised, 1))
# LSupervised = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.matmul(yImages,tf.transpose(yTexts[:numClass])), labels=tf.one_hot(yLabels, numClass)))
# LSupervised = tf.reduce_mean(tf.losses.hinge_loss(logits=tf.matmul(yImages,tf.transpose(yTexts[:numClass])), labels=tf.one_hot(yLabels, numClass)))
# LSupervised = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.matmul(yImages,tf.transpose(yTexts[:numClass])), labels=tf.one_hot(yLabels, numClass)))


checkLabel = tf.multiply(tf.matmul(yImages,tf.transpose(yTexts[:numClass])), tf.one_hot(yLabels, numClass))


# UnSupervised Label loss
binaryDot = tf.matmul(yImagesUnlabel, tf.transpose(yTexts[numClass:]))
maxClass = tf.argmax(binaryDot, 1)
LUnSupUnlab = tf.multiply(binaryDot, tf.one_hot(maxClass, numClassUnLabel))
LUnSupUnlab = -1 * tf.reduce_mean(tf.reduce_sum(LUnSupUnlab, 1))
# LUnSupUnlab = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=binaryDot, labels=tf.one_hot(maxClass, numClassUnLabel)))
# LUnSupUnlab = tf.reduce_mean(tf.losses.hinge_loss(logits=binaryDot, labels=tf.one_hot(maxClass, numClassUnLabel)))


# Predict Label
predict = tf.argmax(tf.matmul(yImages,tf.transpose(yTexts)), 1)


# Combine all loss
LReconstruct = lossV + lossT
LUnsupervised = LReconstruct + 0.1*LUnSupUnlab + 0.1*LMmd
LTotal = LSupervised + 0.0*LUnsupervised


# Optimizer
optimizer = tf.train.AdamOptimizer(1e-3).minimize(LTotal)


np.set_printoptions(threshold=np.nan)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for loop in range(1000):
    tmp, _, totalLoss, lV, lT, lM, lS, lU = sess.run([checkLabel, optimizer, LTotal, lossV, lossT, LMmd, LSupervised, LUnSupUnlab],
                                           feed_dict={xImages:trainXImages[:1500], xImagesUnLabel:trainXImages[1500:2250],
                                                      xTexts:trainXTexts[:150],
                                                      yLabels:trainY[:1500]-1,
                                                      numClass:100, numClassUnLabel:50,
                                                      keepProb:0.7})
    predictTrainClass = sess.run(predict,
                                 feed_dict={xImages: trainXImages[:1500], xTexts: trainXTexts[:100], numClass: 100, keepProb:1.0})
    predictUnLabelTrainClass = sess.run(predict,
                                 feed_dict={xImages: trainXImages[1500:2250], xTexts: trainXTexts[100:150], numClass: 50, keepProb:1.0})
    predictTestClass = sess.run(predict,
                                feed_dict={xImages: testXImages[:1490], xTexts: trainXTexts[:100], numClass: 100, keepProb:1.0})

    trainAcc = np.mean(np.equal(predictTrainClass, trainY[:1500] - 1))
    trainUnAcc = np.mean(np.equal(predictUnLabelTrainClass + 100, trainY[1500:2250] - 1))
    testAcc = np.mean(np.equal(predictTestClass, testY[:1490] - 1))
    print "{0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}, {4:.5f}, {5:.5f}, {6:.5f}, {7:.5f}, {8:.5f}".format(trainAcc, trainUnAcc, testAcc, lV, lT, lM, lS, lU, totalLoss)
    # if loop % 99 == 0 and loop != 0:
    #     print tmp[0]
    #     print tmp[50]




# predictClass = sess.run(predict, feed_dict={xImages:trainXImages[:60], xTexts:trainXTexts})
# print np.array(predictClass).shape
# print predictClass
# print trainY[50:60] - 1