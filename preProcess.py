import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the necessary packages
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.models import Model
import numpy as np
import glob
from gensim.models import Word2Vec, Doc2Vec
from sklearn import preprocessing
import pickle
import cPickle


# Input size and pre process image
inputShape = (299, 299)
preprocess = preprocess_input


# load our the network weights from disk (NOTE: if this is the
# first time you are running this script for a given network, the
# weights will need to be downloaded first -- depending on which
# network you are using, the weights can be 90-575MB, so be
# patient; the weights will be cached and subsequent runs of this
# script will be *much* faster)
print("[INFO] loading {}...".format(InceptionV3))
Network = InceptionV3
baseModel = Network(weights="imagenet")
model = Model(inputs=baseModel.input, outputs=baseModel.get_layer('avg_pool').output)
# model = baseModel


# load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network
images = []
labels = []
print("[INFO] loading and pre-processing image...")
inputFile = open("metadata/train.txt", "r")
for filename in inputFile.readlines():
    tmp = load_img("images/" + filename.rstrip(), target_size=inputShape)
    images.append(img_to_array(tmp))
    labels.append(int(filename[:3]))


# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through the network
# image = np.expand_dims(image, axis=0)
images = np.array(images)
labels = np.array(labels)


# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
images = preprocess(images)


# Get feature of the image
print("[INFO] classifying image with '{}'...".format(InceptionV3))
preds = model.predict(images)
print("[INFO] shape of images feature '{}'...".format(preds.shape))
print("[INFO] shape of images label '{}'...".format(labels.shape))


# # Print classify label
# P = imagenet_utils.decode_predictions(preds)
# # loop over the predictions and display the rank-5 predictions +
# # probabilities to our terminal
# for j in xrange(len(P)):
#     print "#####################"
#     for (i, (imagenetID, label, prob)) in enumerate(P[j]):
#         print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))


# # Get text attribute for each class
# classes = []
# # model = Word2Vec.load("model/word2vec.bin")
# startAlpha = 0.01
# inferEpoch = 1000
# model = Doc2Vec.load("model/doc2vec.bin")
# print model.docvecs.similarity(model.infer_vector("cat"), model.infer_vector("dog"))
#
# print("[INFO] loading and pre-processing text attribute...")
# inputFile = open("metadata/classes.txt", "r")
# for filename in inputFile.readlines():
#     classes.append(model.infer_vector(filename[4:].replace("_", " ").rstrip(), alpha=startAlpha, steps=inferEpoch))
#     # classes.append(model.wv.word_vec(filename[4:].replace("_", " ").rstrip(), True))
# classes = np.array(classes)
# classes = preprocessing.normalize(classes, norm='l2')
# print("[INFO] shape of classes '{}'...".format(classes.shape))


import gensim
model1 = gensim.models.KeyedVectors.load_word2vec_format("./word2vec-master/GoogleNews-vectors-negative300.bin", binary=True)
classes = []
print("[INFO] loading and pre-processing text attribute...")
inputFile = open("metadata/classes.txt", "r")
for filename in inputFile.readlines():
    wordVec = model1[filename[4:].replace("_", " ").rstrip().split()[0]]
    for word in filename[4:].replace("_", " ").rstrip().split()[1:]:
        wordVec += model1[word]
    classes.append(wordVec)
classes = np.array(classes)
classes = preprocessing.normalize(classes, norm='l2')
print("[INFO] shape of classes '{}'...".format(classes.shape))

# All input save in images, labels, and classes
# Image features, labels of each image, and text attributes for each class.
# print preds[0]
# print labels[0]
# print classes[0]

np.save(open("trainXImage.pkl", "wb"), preds)
np.save(open("trainY.pkl", "wb"), labels)
np.save(open("trainXText.pkl", "wb"), classes)


# load the input image using the Keras helper utility while ensuring
# the image is resized to `inputShape`, the required input dimensions
# for the ImageNet pre-trained network
images = []
labels = []
print("[INFO] loading and pre-processing image...")
inputFile = open("metadata/test.txt", "r")
for filename in inputFile.readlines():
    tmp = load_img("images/" + filename.rstrip(), target_size=inputShape)
    images.append(img_to_array(tmp))
    labels.append(int(filename[:3]))


# our input image is now represented as a NumPy array of shape
# (inputShape[0], inputShape[1], 3) however we need to expand the
# dimension by making the shape (1, inputShape[0], inputShape[1], 3)
# so we can pass it through the network
# image = np.expand_dims(image, axis=0)
images = np.array(images)
labels = np.array(labels)


# pre-process the image using the appropriate function based on the
# model that has been loaded (i.e., mean subtraction, scaling, etc.)
images = preprocess(images)


# Get feature of the image
print("[INFO] classifying image with '{}'...".format(InceptionV3))
preds = model.predict(images)
print("[INFO] shape of images feature '{}'...".format(preds.shape))
print("[INFO] shape of images label '{}'...".format(labels.shape))
np.save(open("testXImage.pkl", "wb"), preds)
np.save(open("testY.pkl", "wb"), labels)
