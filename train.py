#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:04:03 2018

@author: rossspencer
"""

'=============== Imports =============='
import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

'=============== Setup =============='
ground_truth = np.load('ground_truth.npy') # 28 x 3 matrix of x,y pairs of red cars
img_array = np.load('data_train.npy')
# the following are hand-labeled non-red-cars
ground_rumor = [
    [2364, 3796, 0],     #5 different greens from trees
    [2379, 3791, 0],
    [2454, 2854, 0],
    [2460, 2857, 0],
    [1463, 3785, 0],
    
    
    [2380, 3945, 0], #some pretty browns from dirt
    [2394, 3940, 0],
    
        
    [1020, 1570, 0], #nice shades of dark grey from asphault
    [1062, 1591, 0], 
    [1221, 1338, 0], #nice grey from tree shadows ofc
    [1040, 1460, 0], #light dirt
        
    [1180, 1430, 0], #white dirt
    [1120, 1290, 0], #white roof
    [1090, 1290, 0], #dark roof shadow
    [1260, 1300, 0], #light grey roof
    [840, 1220, 0], #tan grey roof
    [840, 1200, 0], #light grey roof
        
    [790, 1710, 0], #lighter asphaults
    [750, 1700, 0], 
    [776, 1688, 0], 
    [755, 1685, 0],
    
    [4455, 5730, 0], #light teal roof
    [4343, 5780, 0], #tricky: light brown roof that looks red
    [3630, 880, 0],  #light green golf course color
    [200, 3400, 0],  #swampy brown
    [355, 3200, 0],  #green grass
    [400, 3203, 0],  #brown grass
    [658, 3223, 0],  #tilled dirt grass
    
    [875, 3215, 0], #beautiful puke green tree
    [874, 3215, 0], 
    
    [4323, 4650, 0], #reddish brown dirt
    [620, 6200, 0], #brownish brown dirt lol
        
        
    ## a chunk of random values to see if this changes things, stole them from 1 iteration of the loop to generate ground_truth_rand
    ##this raises accuracy at the expense of finding some red cars 
    [4345, 1095, 0],
    [3586, 5784, 0],
    [5059, 4786, 0],
    [4520, 2523, 0],
    [745, 3705, 0],
    [1399, 1625, 0],
    [3822, 4546, 0],
    [5272, 100, 0],
    [4209, 3690, 0],
    [2119, 5219, 0],
    [4893, 5464, 0],
    [3921, 1580, 0],
    [2753, 12, 0],
    [2444, 5312, 0],
    [2284, 1251, 0],
    [4342, 1329, 0],
    [3209, 4066, 0],
    [1035, 2207, 0],
    [997, 2818, 0],
    [4029, 2976, 0],
    [1745, 1949, 0],
    [3557, 5770, 0],
    [2523, 286, 0],
    [5136, 207, 0],
    [5120, 605, 0],
    [5846, 391, 0],
    [2803, 949, 0],
    [4178, 5725, 0],
    [2610, 5126, 0],
    [2969, 687, 0],
    [4908, 2147, 0],
    [2888, 2482, 0],
    [2047, 3997, 0],
    [2766, 5316, 0],
    [4534, 4832, 0],
    [5141, 750, 0],
    [5461, 347, 0],
    [4934, 657, 0],
    [3173, 5271, 0],
    [5365, 2951, 0],
    [142, 5160, 0],
    [5520, 2143, 0],
    [3645, 6029, 0],
    [4417, 1562, 0],
    [5888, 5505, 0],
    [4052, 2972, 0],
    [3106, 2864, 0],
    [4898, 467, 0],
    [6165, 1409, 0],
    [3818, 3007, 0],
    [4773, 462, 0],
    [1147, 503, 0],
    [5678, 5233, 0],
    [1790, 1486, 0],
    [2123, 1958, 0],
    [2926, 103, 0],
    [887, 3230, 0],
    [3348, 3694, 0],
    [2477, 4387, 0],
    [5118, 4513, 0],
    [3286, 63, 0],
    [4097, 2846, 0],
    [5175, 4730, 0],
    [2331, 4926, 0],
    [5896, 1151, 0],
    [3524, 3236, 0]
        
]

## generate random ground truth with 600 random values (chance one of the ones generated is a red car is near-neglegible), 
## not particularly useful since having a large amount of non-red-cars decreases accuracy since it'll always predict not-red-car
ground_rumor_rand = []
for i in range(600):
    idx = np.random.randint(6250, size=2)
    ground_rumor_rand.append([idx[0], idx[1], 0])


'=============== Parameters =============='

# K is the number of nearest neighbors, M is the random state for test_split for cross-val
K = 4
M = 8

'=============== Function definitions for cross validation =============='

# takes 2 arrays and combines them, converts elements of the 2nd to np arrays since I didn't do that when making my ground_rumor hand-labeled non-red-cars
def combineData(arr1, arr2):
    training_data = []
    for i in range(len(arr1)):
        training_data.append(arr1[i])
    
    for i in range(len(arr2)):
        training_data.append(np.array(arr2[i]))
        
    training_data = np.array(training_data)
    #training_data.shape # 60 rows + x * y * class => 60x3
    return training_data

# splits data for validation then breaks it into different classes
def get_testing_data(arr1, arr2, m):
    training_data = combineData(arr1, arr2)
    Train = [[row[0], row[1]] for row in training_data]
    labels = [[row[2]] for row in training_data]
    X_train_class = []

    #sets random state then splits our data so 33% of it is saved for validation
    M = m
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)

    #X_train_class[0] is others, and X_train_class[1] is red cars
    X_train_class = []  
    red_cars = []
    others = []
    for i in range(len(label_train)):
        if label_train[i] == 2:
            red_cars.append(X_train[i])
        
        else:
            others.append(X_train[i])
    X_train_class.append(others)
    X_train_class.append(red_cars)
    
    return X_train, X_valid, label_train, label_valid, X_train_class

## takes a mask of size 5x5 around each pixel then takes the average RGB value
def avgRGB(coord):
    rgb_vals = []        
    x = np.arange(coord[0] - 2, coord[0] + 3, 1)
    y = np.arange(coord[1] - 2, coord[1] + 3, 1)
    arr = []
    for i in x:
        for j in y:
            ## print(img_array[i][j])
            ### arr will be size 25 consisting of all 1x3 tuples of RGB vals within 5 px square around each red car in the ground truth array
            rgb_vals.append(img_array[i][j])
    
    r = 0
    b = 0
    g = 0
    for i in range(len(rgb_vals)):
        arr = rgb_vals[i]
        r += arr[0]
        b += arr[1]
        g += arr[2]
        
    r = r/len(rgb_vals)
    b = b/len(rgb_vals)
    g = g/len(rgb_vals)
    
    return [r,b,g]


#### implementation of regular old KNN specifically to cross validate our data
def KNN_validation(K):
    # first split data to cross-validate
    X_train, X_valid, label_train, label_valid, X_train_class = get_testing_data(ground_truth, ground_rumor, M)
    n_neighbors = K
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')
    rgb = []
    for x in X_train:
        rgb.append(img_array[x[0]][x[1]][:])
    clf.fit(rgb, np.array(label_train).T[0])
    rgb_valid = []
    for x in X_valid:
        rgb_valid.append(img_array[x[0]][x[1]][:])
    predictions_KNN = clf.predict(rgb_valid)

    #print(accuracy_score(label_valid, predictions_KNN), " ", n_neighbors," nearest neighbors using pixel alone")
    #print(precision_score(label_valid, predictions_KNN, average="macro"))
    #return clf
    return accuracy_score(label_valid, predictions_KNN)
    #return precision_score(label_valid, predictions_KNN, average="micro")

#### implementation of KNN utilizing masked averaging specifically to cross validate our data
def KNN_avg_validation(K):
    X_train, X_valid, label_train, label_valid, X_train_class = get_testing_data(ground_truth, ground_rumor, M)
    n_neighbors = K
    clf_masked = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    rgb_masked = []
    for x in X_train:
        rgb_masked.append(avgRGB(x))

    clf_masked.fit(rgb_masked, np.array(label_train).T[0])

    rgb_masked_valid = []
    for x in X_valid:
        ## append the average RGB around x rather than the RGB at x itself
        rgb_masked_valid.append(avgRGB(x))

    predictions_KNN_masked = clf_masked.predict(rgb_masked_valid)
    
    #print(accuracy_score(label_valid, predictions_KNN_masked), " ", n_neighbors," nearest neighbors using masked avg")
    #print(precision_score(label_valid, predictions_KNN_masked, average="macro"))
    #return clf_masked
    return accuracy_score(label_valid, predictions_KNN_masked)
    #return precision_score(label_valid, predictions_KNN_masked, average="micro")
    
#    
##
#KNN_validation(4)
#KNN_avg_validation(4)
#arr = np.zeros(shape=(12, 2, 20))
#for M in range(1,13):
#    #print("random state: ", M)
#    for i in range(1,21):
#        arr[M-1][0][i-1] = KNN_validation(i)
#        arr[M-1][1][i-1] = KNN_avg_validation(i)
#        
##
##for m in range(1,13):
##    plt.close('all')
##    # for M=k, the accuracy score plotted gives us 
##    plt.title("Random State M = "+str(m))
##    plt.xlabel("Number of Nearest Neighbors")
##    plt.ylabel("Accuracy Score")
##    plt.plot([i for i in range(1,21)], arr[m-1][0], 'orange', label='Without Masked Averaging') #without masking
##    plt.plot([i for i in range(1,21)], arr[m-1][1], 'b', label='With Masked Averaging') #with masking
##    plt.legend()
##   
##   
##    plt.show()
##    
#    
#for m in range(1,13):
#    plt.close('all')
#    # for M=k, the accuracy score plotted gives us 
#    plt.title("Random State M = "+str(m))
#    plt.xlabel("Number of Nearest Neighbors")
#    plt.ylabel("Precision Score")
#    plt.plot([i for i in range(1,21)], arr[m-1][0], 'orange', label='Without Masked Averaging') #without masking
#    plt.plot([i for i in range(1,21)], arr[m-1][1], 'b', label='With Masked Averaging') #with masking
#    plt.legend()
#    
#    
#    plt.show()
##    



'=============== Function definitions for actual training =============='
#
def get_training_data(arr1, arr2):
    training_data = combineData(arr1, arr2)
    Train = [[row[0], row[1]] for row in training_data]
    labels = [row[2] for row in training_data]
    X_train_class = []

    red_cars = []
    others = []
    for i in range(len(labels)):
        if labels[i] == 2:
            red_cars.append(Train[i])
        
        else:
            others.append(Train[i])
            
    X_train_class.append(others)
    X_train_class.append(red_cars)
    
    X_train_class = np.array(X_train_class)
    
    return X_train_class

#KNN that has 1 already-merged array passed in with labels 
def KNN_dataset(data, N):
    n_neighbors = N
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    rgb = []
    for x in data:
        rgb.append(img_array[x[0]][x[1]][:])

    clf.fit(rgb, data[:,[2]].T[0])
    return clf
    
#KNN with averaging that has 1 already-merged array passed in with labels 
def KNN_avg_dataset(data, N):
    n_neighbors = N
    clf_masked = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    rgb_masked = []
    for x in data:
        rgb_masked.append(avgRGB(x))    
    clf_masked.fit(rgb_masked, data[:,[2]].T[0])
    
    return clf_masked


#KNN that takes in 2 datasets and merges them to train the classifier
def KNN(arr1, arr2, N):
    data = combineData(arr1, arr2)
    return KNN_dataset(data, N)


#KNN using pixel rgb averaging that takes in 2 datasets and merges them to train the classifier
def KNN_avg(arr1, arr2, N):
    data = combineData(arr1, arr2)
    return KNN_avg_dataset(data, N)

' The actual train function, takes in a dataset and trains classifier using it '
def train(X):
    #clf2 = KNN_avg_dataset(X, 3)
    clf2 = KNN_dataset(X, 3)
    return clf2

#finds all red cars in the window x in (x1,x2) by y in (y1,y2)
    'a *function* that will run your training code on an input data set X and return the location of red cars in Y. '
    'The output should be a numpy array containing the row- and column- location of each car. '
def findAllRedCars(Y, X, x1, x2, y1, y2, stepsize):
    clf_trained = train(X)
    coords = []
    testingArr = []
    for i in range(x1, x2, stepsize):
        for j in range(y1, y2, stepsize):
            coords.append([i,j])
            testingArr.append(Y[i, j])

    predictions = clf_trained.predict(testingArr)
    
##### this code is used to generate plots of the predicted red cars against the actual image:
    plt.imshow(Y)
    for i in range(len(predictions)):
        if predictions[i] == 2:
                plt.plot([coords[i][0]], [coords[i][1]],'o')
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.show()
    
##### this code would makes an array of the predicted red cars
    red_cars = []
    for i in range(len(predictions)):
        if predictions[i] == 2:
            red_cars.append(coords[i])
    return red_cars


########################## running test cases to generate plots for report: 
# data = combineData(ground_truth, ground_rumor)
# print(findAllRedCars(img_array, data, 4400, 4500, 5400, 5600, 15))
