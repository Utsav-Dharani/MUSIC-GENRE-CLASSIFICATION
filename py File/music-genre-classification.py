# Importing Libraries: 

from python_speech_features import mfcc # Mel frequency Cepstral Coefficients
import scipy.io.wavfile as wav 
import numpy as np

from tempfile import TemporaryFile

import os
import pickle
import random
import operator

import math

# Function to perform actual distance calculation between features
def distance(instance1, instance2, k):
    
    distance = 0
    mm1 = instance1[0]
    cm1 = instance1[1]
    mm2 = instance1[0]
    cm2 = instance1[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1))
    distance += (np.dot(np.dot((mm2 - mm1).transpose(), np.linalg.inv(cm2)), mm2 - mm1 ))
    distance += np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance -= k
    return distance
    
# Function to get the distance between features vectors and find neighbour
def getNeighbors(trainingSet, instance, k):
    
    distances = []
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
        
    distances.sort(key=operator.itemgetter(1))
    
    neighbors = []
    for x in range (k):
        neighbors.append(distances[x][0])
    
    return neighbors

# Identifiy the class of neighbors
def nearestclass(neighbors):
    
    classVote = {}
    
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response] += 1
            
        else:
            classVote[response] = 1
            
    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse = True)
    
    return sorter[0][0]

# Function to evaluate the model
def getAccuracy(testSet, prediction):
    
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == prediction[x]:
            correct += 1
        
    return (1.0 * correct) / len(testSet)

# Directory that holds the dataset
directory = 'F:/MUSIC GENRE CLASSIFICATION/Data/genres_original/'

# Binary file where we will collect all the features extracted using mfcc (mel frequency cepstral conefficients)

f = open("VLC media player", 'wb')

i = 0

for folder in os.listdir(directory):
    
    i += 1
    
    if i == 11:
        break
    
    for file in os.listdir(directory + folder):
        try:
            (rate, sig) = wav.read(directory + folder + "/" + file)
            mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy = False)
            covariance = np.cov(np.matrix.transpose(mfcc_feat))
            mean_matrix = mfcc_feat.mean(0)
            feature = (mean_matrix, covariance, i)
            pickle.dump(feature, f)
        
        except Exception as e:
            print('Got an exception: ', e, 'in folder: ', folder, 'filename: ', file)
        
f.close()
        
# Split the dataset into training and testing sets respectively
dataset = []
def loadDataset(filename, split, trSet, teSet):
    
    with open(filename, 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            
            except EOFError:
                f.close()
                break
    
    for x in range(len(dataset)):
        
        if random.random() < split:
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])
            
trainingSet = []
testSet = []

loadDataset("VLC media player", 0.66, trainingSet, testSet)

# Making predection using KNN
leng = len(testSet)
predections = []

for x in range(leng):
    predections.append(nearestclass(getNeighbors(trainingSet, testSet[x], 5)))
    
accuracy1 = getAccuracy(testSet, predections)
print(accuracy1)            
            
from collections import defaultdict
results = defaultdict(int)

i = 1

for folder in os.listdir(directory):
    
    results[i] = folder
    i += 1
    
    print(results)
    
# Testing the code with external sample
# URL: https://uweb.engr.arizona.edu/-429rns/audiofiles/audiofiles.html

test_dir = "F:/MUSIC GENRE CLASSIFICATION/Test/"
#test_file = test_dir + "test.wav"
test_file = test_dir + "test2.wav"
# test_file = test_dir + "test4.wav"    
            
(rate, sig) = wav.read(test_file)
mfcc_feat = mfcc(sig, rate, winlen = 0.020, appendEnergy = False)
covariance = np.cov(np.matrix.transpose(mfcc_feat))
mean_matrix = mfcc_feat.mean(0)
features = (mean_matrix, covariance, i)

pred = nearestclass(getNeighbors(dataset, feature, 5))
print(results[pred])
