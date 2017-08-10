import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the necessary packages
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.datasets import cifar100
from sklearn import preprocessing
import numpy as np
import scipy
from scipy import misc
import cPickle
import gensim

meta = cPickle.load(open("./cifar100/meta", "rb"))
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print meta['fine_label_names'][0]
print y_train[0][0]


# Inception Model
baseModel = InceptionV3(weights="imagenet")
model = Model(inputs=baseModel.input, outputs=baseModel.get_layer('avg_pool').output)

# big_x_train = np.array([scipy.misc.imresize(x_train[i], (299, 299, 3)) for i in range(0, len(x_train))]).astype('float32')
big_x_test = np.array([scipy.misc.imresize(x_train[i], (299, 299, 3)) for i in range(0, len(x_train))]).astype('float32')

# predsTrain = model.predict(preprocess_input(big_x_train))
predsTest = model.predict(preprocess_input(big_x_test))


# Word to Vec Model
modelWord = gensim.models.KeyedVectors.load_word2vec_format("./word2vec-master/GoogleNews-vectors-negative300.bin", binary=True)
classes = []
for wordClass in meta['fine_label_names']:
    wordSplit = wordClass.replace("_", " ").rstrip().split()
    wordVec = modelWord[wordSplit[0]]
    for word in wordSplit[1:]:
        wordVec += modelWord[word]
    classes.append(wordVec)
classes = np.array(classes)
classes = preprocessing.normalize(classes, norm='l2')

print  predsTest.shape, classes.shape

mapping = dict()
# trainClass = ["beaver", "dolphin", "otter", "aquarium_fish", "flatfish", "orchid", "poppy", "rose", "bottle", "bowl",
#               "apple", "mushroom", "orange", "clock", "keyboard", "bed", "chair", "couch", "bee", "beetle",
#               "bear", "leopard", "lion", "bridge", "castle", "cloud", "forest", "mountain", "camel", "cattle",
#               "fox", "porcupine", "possum", "crab", "lobster", "snail", "baby", "boy", "crocodile", "dinosaur", "lizard",
#               "hamster", "mouse", "maple_tree", "oak_tree", "palm_tree", "bicycle", "bus", "lawn_mower", "rocket",
#               "seal", "ray", "shark", "sunflower", "can", "cup", "sweet_pepper", "telephone", "wardrobe",
#               "butterfly", "caterpillar", "tiger", "road", "skyscraper", "plain", "chimpanzee", "elephant",
#               "raccoon", "worm", "man", "woman", "turtle", "rabbit", "shrew", "pine_tree", "motorcycle", "pickup_truck",
#               "tank", "tractor", "lamp"]
trainClass = ["beaver", "dolphin", "otter", "aquarium_fish", "orchid", "poppy", "rose", "bottle", "bowl",
              "apple", "mushroom", "clock", "keyboard", "bed", "chair", "couch", "bee", "beetle",
              "bear", "leopard", "lion", "bridge", "castle", "cloud", "mountain", "camel", "cattle",
              "fox", "porcupine", "possum", "crab", "snail", "baby", "crocodile", "dinosaur", "lizard",
              "hamster", "mouse", "maple_tree", "oak_tree", "palm_tree", "bicycle", "bus", "lawn_mower", "rocket",
              "seal", "ray", "shark", "sunflower", "can", "cup", "sweet_pepper", "telephone", "wardrobe",
              "butterfly", "caterpillar", "tiger", "road", "skyscraper", "plain", "chimpanzee", "elephant",
              "raccoon", "worm", "man", "woman", "turtle", "rabbit", "shrew", "pine_tree", "motorcycle",
              "tank", "tractor", "lamp",
              "whale", "trout", "tulip", "plate",
              "pear", "television", "table", "cockroach",
              "wolf", "house", "sea", "kangaroo",
              "skunk", "spider", "girl", "snake", "squirrel",
              "willow_tree", "train", "streetcar"]
trainX = []
trainY = []
trainZ = []
for _, className in enumerate(trainClass):
    index = meta['fine_label_names'].index(className)
    trainZ.append(classes[index])
    mapping[index] = _

# testClass = ["whale", "trout", "tulip", "plate",
#              "pear", "television", "table", "cockroach",
#              "wolf", "house", "sea", "kangaroo",
#              "skunk", "spider", "girl", "snake", "squirrel",
#              "willow_tree", "train", "streetcar"]
testClass = ["forest", "lobster", "orange", "boy", "pickup_truck", "flatfish"]
testX = []
testY = []
testZ = []
for _, className in enumerate(testClass):
    index = meta['fine_label_names'].index(className)
    testZ.append(classes[index])
    mapping[index] = _

for l in range(y_train.shape[0]):
    if meta['fine_label_names'][y_train[l][0]] in trainClass:
        trainX.append(predsTest[l])
        trainY.append(mapping[y_train[l][0]])
    else:
        testX.append(predsTest[l])
        testY.append(mapping[y_train[l][0]])

trainX = np.array(trainX)
trainY = np.array(trainY)
trainZ = np.array(trainZ)
testX = np.array(testX)
testY = np.array(testY)
testZ = np.array(testZ)

print mapping
print trainX.shape, trainY.shape, trainZ.shape, testX.shape, testY.shape, testZ.shape

np.save(open("trainX100.pkl", "wb"), trainX)
np.save(open("trainY100.pkl", "wb"), trainY)
np.save(open("trainZ100.pkl", "wb"), trainZ)
np.save(open("testX100.pkl", "wb"), testX)
np.save(open("testY100.pkl", "wb"), testY)
np.save(open("testZ100.pkl", "wb"), testZ)
