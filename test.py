#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 11:04:16 2018

@author: rossspencer
"""

'=============== Imports =============='
import numpy as np
from matplotlib import pyplot as plt
import train as trainingFunctions

'a *function* that will run your testing code on an input data set X. Your test.py code should already be trained'
def findAllRedCarsTest(Y, X, x1, x2, y1, y2, stepsize):
    clf_trained = trainingFunctions.train(X)
    coords = []
    testingArr = []
    for i in range(x1, x2, stepsize):
        for j in range(y1, y2, stepsize):
            coords.append([i,j])
            testingArr.append(Y[i, j])
    
    predictions = clf_trained.predict(testingArr)
    
##### this code is used to generate plots of the predicted red cars against the actual test image:
    plt.imshow(Y)
    for i in range(len(predictions)):
        if predictions[i] == 2:
                plt.plot([coords[i][0]], [coords[i][1]],'o')
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)
    plt.show()
    
    red_cars = []
    for i in range(len(predictions)):
        if predictions[i] == 2:
            red_cars.append(coords[i])
    return red_cars

