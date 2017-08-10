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

keepProb = tf.placeholder(tf.float32)

# xImages = tf.placeholder(tf.float32, [None, 2048], name='XImages_placeholder')
# w1_1 = tf.Variable(tf.random_normal([2048, 500]))
# b1_1 = tf.Variable(tf.random_normal([500]))
# o1_1 = tf.nn.tanh(tf.add(tf.matmul(xImages, w1_1), b1_1))
# w2_1 = tf.Variable(tf.random_normal([500, 100]))
# b2_1 = tf.Variable(tf.random_normal([100]))
# o2_1 = tf.nn.tanh(tf.add(tf.matmul(o1_1, w2_1), b2_1))
# w3_1 = tf.Variable(tf.random_normal([100, 50]))
# b3_1 = tf.Variable(tf.random_normal([50]))
# output = tf.add(tf.matmul(o2_1, w3_1), b3_1)
# output = tf.nn.dropout(output, keepProb)
# output = tf.nn.l2_normalize(output, 1, epsilon=1e-12)
#
#
# xTexts = tf.placeholder(tf.float32, [None, 300], name='XTexts_placeholder')
# w1_2 = tf.Variable(tf.random_normal([300, 100]))
# b1_2 = tf.Variable(tf.random_normal([100]))
# o1_2 = tf.nn.tanh(tf.add(tf.matmul(xTexts, w1_2), b1_2))
# w2_2 = tf.Variable(tf.random_normal([100, 50]))
# b2_2 = tf.Variable(tf.random_normal([50]))
# output1 = tf.add(tf.matmul(o1_2, w2_2), b2_2)
# output1 = tf.nn.dropout(output1, keepProb)
# output1 = tf.nn.l2_normalize(output1, 1, epsilon=1e-12)


xImages = tf.placeholder(tf.float32, [None, 2048], name='XImages_placeholder')
w1_1 = tf.Variable(tf.random_normal([2048, 200]))
b1_1 = tf.Variable(tf.random_normal([200]))
output = tf.add(tf.matmul(xImages, w1_1), b1_1)
output = tf.nn.dropout(output, keepProb)
output = tf.nn.l2_normalize(output, 1, epsilon=1e-12)


xTexts = tf.placeholder(tf.float32, [None, 300], name='XTexts_placeholder')
w1_2 = tf.Variable(tf.random_normal([300, 200]))
b1_2 = tf.Variable(tf.random_normal([200]))
output1 = tf.add(tf.matmul(xTexts, w1_2), b1_2)
output1 = tf.nn.dropout(output1, keepProb)
output1 = tf.nn.l2_normalize(output1, 1, epsilon=1e-12)


yLabels = tf.placeholder(tf.int32, [None], name='YLabels_placeholder')
oneHotClass = tf.placeholder(tf.int32)
dotProduct = tf.matmul(output, tf.transpose(output1))
# entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dotProduct, labels=tf.one_hot(yLabels, oneHotClass)))
entropy = -1 * tf.reduce_mean(tf.reduce_sum(tf.multiply(dotProduct, tf.one_hot(yLabels, oneHotClass)), 1))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(entropy)


predict = tf.argmax(dotProduct, 1)


choice = np.arange(200)
# np.random.shuffle(choice)

trainXImagesTmp = trainXImages[choice[0]*15:(choice[0]*15)+15]
trainYTmp = trainY[choice[0]*15:(choice[0]*15)+15] - choice[0] - 1
trainXTextsTmp = [trainXTexts[choice[0]]]
count = 1
for index in choice[1:100]:
    trainXImagesTmp = np.concatenate((trainXImagesTmp, trainXImages[index*15:(index*15)+15]), axis=0)
    trainYTmp = np.concatenate((trainYTmp, trainY[index*15:(index*15)+15] - index - 1 + count), axis=0)
    trainXTextsTmp = np.concatenate((trainXTextsTmp, [trainXTexts[index]]), axis=0)
    count += 1


disTestX = trainXImages[choice[100]*15:(choice[100]*15)+15]
disTestY = trainY[choice[100]*15:(choice[100]*15)+15] - choice[100] - 1
disText = [trainXTexts[choice[100]]]
count = 1
for index in choice[101:150]:
    disTestX = np.concatenate((disTestX, trainXImages[index*15:(index*15)+15]), axis=0)
    disTestY = np.concatenate((disTestY, trainY[index*15:(index*15)+15] - index - 1 + count), axis=0)
    disText = np.concatenate((disText, [trainXTexts[index]]), axis=0)
    count += 1


# # Disjoint class
# disTestX = trainXImages[1500:1530]
# disTestY = trainY[1500:1530]
#
#
# # Split data
# trainXImages = trainXImages[:1500]
# trainY = trainY[:1500]
testXImages = testXImages[:1490]
testY = testY[:1490]
numClasses = 100


np.set_printoptions(threshold=np.nan)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
miniBatchSize = 100
for loop in range(10000):
    losses = []
    for i in range(0, trainXImagesTmp.shape[0], miniBatchSize):
        xBatch = trainXImagesTmp[i:i + miniBatchSize]
        yBatch = trainYTmp[i:i + miniBatchSize]
        _, loss = sess.run([optimizer, entropy], feed_dict={xImages:xBatch, yLabels:yBatch, xTexts:trainXTextsTmp, oneHotClass:100, keepProb:1.0})
        losses.append(loss)

    totalLoss = sum(losses) / len(losses)
    predictTrainClass = sess.run(predict, feed_dict={xImages:trainXImagesTmp, xTexts:trainXTextsTmp, keepProb:1.0})
    predictTestClass = sess.run(predict, feed_dict={xImages:testXImages, xTexts:trainXTexts[:numClasses], keepProb:1.0})
    predictDisClass = sess.run(predict, feed_dict={xImages:disTestX, xTexts:disText, keepProb:1.0})

    trainAcc = np.mean(np.equal(predictTrainClass, trainYTmp))
    testAcc = np.mean(np.equal(predictTestClass, testY - 1))
    disAcc = np.mean(np.equal(predictDisClass, disTestY))
    print "{0:.5f}, {1:.5f}, {2:.5f}, {3:.5f}".format(trainAcc, testAcc, disAcc, totalLoss)