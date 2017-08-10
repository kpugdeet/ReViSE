import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the necessary packages
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.datasets import cifar10
from sklearn import preprocessing
import numpy as np
import scipy
from scipy import misc
import cPickle
import gensim

meta = cPickle.load(open("./cifar10/batches.meta", "rb"))
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print meta['label_names'][0]
print y_train[0][0]


# Inception Model
baseModel = InceptionV3(weights="imagenet")
model = Model(inputs=baseModel.input, outputs=baseModel.get_layer('avg_pool').output)

# big_x_train = np.array([scipy.misc.imresize(x_train[i], (299, 299, 3)) for i in range(0, len(x_train))]).astype('float32')
big_x_test = np.array([scipy.misc.imresize(x_test[i], (299, 299, 3)) for i in range(0, len(x_test))]).astype('float32')

# predsTrain = model.predict(preprocess_input(big_x_train))
predsTest = model.predict(preprocess_input(big_x_test))


# Word to Vec Model
modelWord = gensim.models.KeyedVectors.load_word2vec_format("./word2vec-master/GoogleNews-vectors-negative300.bin", binary=True)
classes = []
for wordClass in meta['label_names']:
    classes.append(modelWord[wordClass])
classes = np.array(classes)
classes = preprocessing.normalize(classes, norm='l2')

print  predsTest.shape, classes.shape

trainX = []
trainY = []
testX = []
testY = []
for l in range(y_test.shape[0]):
    if y_test[l][0] < 5:
        trainX.append(predsTest[l])
        trainY.append(y_test[l][0])
    else:
        testX.append(predsTest[l])
        testY.append(y_test[l][0] - 5)

trainX = np.array(trainX)
trainY = np.array(trainY)
trainZ = np.array(classes[:5])
testX = np.array(testX)
testY = np.array(testY)
testZ = np.array(classes[5:])

print trainX.shape, trainY.shape, trainZ.shape, testX.shape, testY.shape, testZ.shape

np.save(open("trainX.pkl", "wb"), trainX)
np.save(open("trainY.pkl", "wb"), trainY)
np.save(open("trainZ.pkl", "wb"), trainZ)
np.save(open("testX.pkl", "wb"), testX)
np.save(open("testY.pkl", "wb"), testY)
np.save(open("testZ.pkl", "wb"), testZ)
