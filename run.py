#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 19:26:10 2018

@author: rossspencer
"""


'A run.py script that runs your training and testing functions on a small sample image (this script should not take longer than approximately 5 minutes to run)'

import train as trainingFunctions
import test as testingFunctions
import numpy as np

data = trainingFunctions.combineData(trainingFunctions.ground_truth, trainingFunctions.ground_rumor)
image_test = np.load('data_test.npy')
print(testingFunctions.findAllRedCarsTest(image_test, data, 1000, 2000, 5300, 5600, 15))

## to change KNN averaged to regular KNN, switch which line is commented out in train(X) in train.py