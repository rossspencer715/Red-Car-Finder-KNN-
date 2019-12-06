#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 17:41:32 2018

@author: rossspencer
"""


##### SUPER messy prototype work-file, please disregard


import numpy as np
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from matplotlib import pyplot as plt

ground_truth = np.load('ground_truth.npy')
np.array(ground_truth)[0]
#print(ground_truth[0])     #28 x 3 matrix
#np.array(ground_truth).size

data = np.random.normal(0, 1, 100)
np.save('data.npy', data)

data = np.load('data.npy')
data.shape
data[0]
np.zeros(data.shape)
img = plt.scatter(data, np.zeros(data.shape), marker='^')
plt.show(img)




img_array = np.load('data_train.npy')
#print(img_array.shape)

# shows the green channel
#plt.imshow(img_array[:,:,2])
#plt.show()

# zooms in to a window of x in 1000x1200 by y in 1500x1700
#plt.imshow(img_array[:,:,:])
#plt.xlim(1000, 1200)
#plt.ylim(1500, 1700)
#plt.show()

#### plots the ground truth cars in 5x5 windows, cars look pretty much centered
#for i in range(len(ground_truth)):
#    plt.imshow(img_array)
#    plt.xlim(ground_truth[i][0]-2, ground_truth[i][0]+2)
#    plt.ylim(ground_truth[i][1]-2, ground_truth[i][1]+2)
#    plt.show()
    
#### looking at the ground truth with more red cars from Canvas discussion, they don't appear very well centered. I'm not going to use this file.
#ground_truth2 = np.load('more_red_cars.npy')
#for i in range(len(ground_truth2)):
#    plt.imshow(img_array)
#    plt.xlim(ground_truth2[i][0] - 15, ground_truth2[i][0] + 15)
#    plt.ylim(ground_truth2[i][1] - 15, ground_truth2[i][1] + 15)
#    plt.show()
    
#### peaking at some rgb values in the ground truth 
#print(img_array[903,1186])
#print(img_array[1112,1558])
#print(img_array[1767,1361])

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
        [620, 6200, 0] #brownish brown dirt lol
        
]

ground_rumor_rand = []
for i in range(600):
    idx = np.random.randint(6250, size=2)
    ground_rumor_rand.append([idx[0], idx[1], 0])

##### plots the ground rumor aka non-cars in 5x5 windows
#for i in range(len(ground_rumor)):
#    plt.imshow(img_array)
#    plt.xlim(ground_rumor[i][0]-2, ground_rumor[i][0]+3)
#    plt.ylim(ground_rumor[i][1]-2, ground_rumor[i][1]+3)
#    plt.show()
    
def combineData(arr1, arr2):
    training_data = []
    for i in range(len(arr1)):
        training_data.append(arr1[i])
    
    for i in range(len(arr2)):
        training_data.append(np.array(arr2[i]))
        
    training_data = np.array(training_data)
    #training_data.shape # 60 rows + x * y * class => 60x3
    return training_data


from sklearn.model_selection import train_test_split

def get_testing_data(arr1, arr2, m):
    training_data = combineData(arr1, arr2)
    Train = [[row[0], row[1]] for row in training_data]
    labels = [[row[2]] for row in training_data]
    #Classes = np.sort(np.unique(labels))
    X_train_class = []

    #sets random state then splits our data so 33% of it is saved for validation
    M = m
    X_train, X_valid, label_train, label_valid = train_test_split(Train, labels, test_size = 0.33, random_state = M)


    X_train_class = []
    #breaks our training data into classes so that the ith class is in X_train_class[i-1]
    #for j in range(Classes.shape[0]):
    #    jth_class = X_train[label_train == Classes[j],:]#,:]
    #    X_train_class.append(jth_class)
    #    

    red_cars = []
    others = []
    for i in range(len(label_train)):
        if label_train[i] == 2:
            red_cars.append(X_train[i])
        
        else:
            others.append(X_train[i])
            
    X_train_class.append(others)
    X_train_class.append(red_cars)

    #print(X_train_class[0]) #ground_rumor non-red-cars
    #print(X_train_class[1]) #ground_truth cars
    
    return X_train, X_valid, label_train, label_valid, X_train_class

N = 2
M = 12
#### implementation of KNN 
def KNN_validation(N):
    #for i in range(1, 25):
    X_train, X_valid, label_train, label_valid, X_train_class = get_testing_data(ground_truth, ground_rumor_rand, M)
    n_neighbors = N
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    rgb = []
    for x in X_train:
        rgb.append(img_array[x[0]][x[1]][:])

    clf.fit(rgb, np.array(label_train).T[0])
    

    rgb_valid = []
    for x in X_valid:
        rgb_valid.append(img_array[x[0]][x[1]][:])


    predictions_KNN = clf.predict(rgb_valid)


    

    print(accuracy_score(label_valid, predictions_KNN), " ", n_neighbors," nearest neighbors using pixel alone")
    print(precision_score(np.array(label_valid).T[0], predictions_KNN, pos_label=2), " precision score")
    #return clf
    return accuracy_score(label_valid, predictions_KNN)


KNN_validation(20)



## gets mask of size 5x5 mask around each pixel?? an idea maybe???
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


## print(avgRGB([903,1186]))
#x = np.arange(903 - 2, 903 + 3, 1)
#y = np.arange(1186 - 2, 1186 + 3, 1)
#
##### peaking at some rgb values
#print(img_array[903,1186,0])
#print(img_array[903,1186,1])
#print(img_array[903,1186,2])
#
#print(img_array[1112,1558,0])
#print(img_array[1112,1558,1])
#print(img_array[1112,1558,2])
#
#print(img_array[1767,1361,0])
#print(img_array[1767,1361,1])
#print(img_array[1767,1361,2])
    
def KNN_avg_validation(N):
    X_train, X_valid, label_train, label_valid, X_train_class = get_testing_data(ground_truth, ground_rumor, M)
#for i in range(1, 25):
    n_neighbors = N
    clf_masked = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    rgb_masked = []
    for x in X_train:
        rgb_masked.append(avgRGB(x))

    clf_masked.fit(rgb_masked, np.array(label_train).T[0])

    rgb_masked_valid = []
    for x in X_valid:
        rgb_masked_valid.append(avgRGB(x))


    predictions_KNN_masked = clf_masked.predict(rgb_masked_valid)


    

    #print(accuracy_score(label_valid, predictions_KNN_masked), " ", n_neighbors," nearest neighbors using masked avg")
    #return clf_masked
    return accuracy_score(label_valid, predictions_KNN_masked)



arr = np.zeros(shape=(12, 2, 20))
for M in range(1,13):
    #print("random state: ", M)
    for i in range(1,21):
        arr[M-1][0][i-1] = KNN_validation(i)
        arr[M-1][1][i-1] = KNN_avg_validation(i)
        
plt.close('all')
# for M=12, the accuracy score plotted gives us 
plt.plot([i for i in range(1,21)], arr[11][0], 'r') #without masking
plt.plot([i for i in range(1,21)], arr[11][1], 'b') #with masking
plt.show()

# for M=10, the accuracy score plotted gives us 
plt.plot([i for i in range(1,21)], arr[9][0], 'r') #without masking
plt.plot([i for i in range(1,21)], arr[9][1], 'b') #with masking
plt.show()

# for M=5, the accuracy score plotted gives us 
plt.plot([i for i in range(1,21)], arr[4][0], 'r') #without masking
plt.plot([i for i in range(1,21)], arr[4][1], 'b') #with masking
plt.show()

for k in range(1,13):
    plt.close('all')
    # for M=k, the accuracy score plotted gives us 
    plt.title("Random State M = "+str(k))
    plt.xlabel("Number of Nearest Neighbors")
    plt.ylabel("Accuracy Score")
    plt.plot([i for i in range(1,21)], arr[k-1][0], 'orange', label='Without Masked Averaging') #without masking
    plt.plot([i for i in range(1,21)], arr[k-1][1], 'b', label='With Masked Averaging') #with masking
    plt.legend()
    
    
    plt.show()
    
    
def get_training_data(arr1, arr2):
    training_data = combineData(arr1, arr2)
    Train = [[row[0], row[1]] for row in training_data]
    labels = [row[2] for row in training_data]
    #Classes = np.sort(np.unique(labels))
    X_train_class = []

    #breaks our training data into classes so that the ith class is in X_train_class[i-1]
    #for j in range(Classes.shape[0]):
    #    jth_class = X_train[label_train == Classes[j],:]#,:]
    #    X_train_class.append(jth_class)
    #    

    red_cars = []
    others = []
    for i in range(len(labels)):
        if labels[i] == 2:
            red_cars.append(Train[i])
        
        else:
            others.append(Train[i])
            
    X_train_class.append(others)
    X_train_class.append(red_cars)

    #print(X_train_class[0]) #ground_rumor non-red-cars
    #print(X_train_class[1]) #ground_truth cars
    
    X_train_class = np.array(X_train_class)
    
    return X_train_class


training_data = combineData(ground_truth, ground_rumor_rand)
training_data[:, [2]]
Train = [[row[0], row[1]] for row in training_data]
labels = [row[2] for row in training_data]

red_cars = []
others = []
for i in range(len(labels)):
    if labels[i] == 2:
        red_cars.append(Train[i])
        
    else:
        others.append(Train[i])
        
def KNN(arr1, arr2, N):
#for i in range(1, 25):
    data = combineData(arr1, arr2)
    n_neighbors = N
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    rgb = []
    for x in data:
        rgb.append(img_array[x[0]][x[1]][:])

    clf.fit(rgb, data[:,[2]].T[0])
    
#
#    rgb_valid = []
#    for x in X_valid:
#        rgb_valid.append(img_array[x[0]][x[1]][:])
#

    #predictions_KNN = clf.predict(rgb_valid)


    

    #print(accuracy_score(label_valid, predictions_KNN), " ", n_neighbors," nearest neighbors using pixel alone")
    return clf
    #return accuracy_score(label_valid, predictions_KNN)

def KNN_dataset(data, N):
#for i in range(1, 25):
    
    n_neighbors = N
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    rgb = []
    for x in data:
        rgb.append(img_array[x[0]][x[1]][:])

    clf.fit(rgb, data[:,[2]].T[0])
    
#
#    rgb_valid = []
#    for x in X_valid:
#        rgb_valid.append(img_array[x[0]][x[1]][:])
#

    #predictions_KNN = clf.predict(rgb_valid)


    

    #print(accuracy_score(label_valid, predictions_KNN), " ", n_neighbors," nearest neighbors using pixel alone")
    return clf
    #return accuracy_score(label_valid, predictions_KNN)
    
    
    
def KNN_avg_dataset(data, N):
    #X_train, X_valid, label_train, label_valid, X_train_class = get_testing_data(ground_truth, ground_rumor, M)
#for i in range(1, 25):

    n_neighbors = N
    clf_masked = neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    rgb_masked = []
    for x in data:
        rgb_masked.append(avgRGB(x))

    
    clf_masked.fit(rgb_masked, data[:,[2]].T[0])



    

    #print(accuracy_score(label_valid, predictions_KNN_masked), " ", n_neighbors," nearest neighbors using masked avg")
    return clf_masked
    #return accuracy_score(label_valid, predictions_KNN_masked)

data = combineData(ground_truth, ground_rumor_rand)
clf2 = KNN_avg_dataset(data, 5)


def train(X):
    clf2 = KNN_dataset(X, 4)
    return clf2

clf_trained = train(data)
img_array.shape[2]
for i in range(2, img_array.shape[0], 5):
    for j in range(237, img_array.shape[1], 5):
        print(i,j)
        
##data starts at x=0, y=237
        
        
        
## let's test on 1800-1100 y and 2200-3000
        
##make array with all of these coordinates, then test clf2 on the array
        
        
test_image = np.load('data_test.npy')
plt.imshow(test_image)
plt.show()

coords = []
testingArr = []
for i in range(2200, 2500, 25):
    for j in range(1100, 1300, 25):
        coords.append([i,j])
        testingArr.append(test_image[i, j])

idk = clf_trained.predict(testingArr)
    
test_image = np.load('data_test.npy')
plt.imshow(test_image)
plt.plot([100,200,300],[200,150,200],'o')   #[x1, x2, x3], [y1, y2, y3]
plt.xlim(50, 400)
plt.ylim(50, 400)
plt.show()

test_image = np.load('data_test.npy')
plt.imshow(test_image)
for i in range(len(idk)):
    if idk[i] == 2:
            plt.plot([coords[i][0]], [coords[i][1]],'o')
plt.xlim(2200, 3000)
plt.ylim(1100, 1800)
plt.show()


for x in ground_truth:
    plt.imshow(img_array)
    plt.xlim(x[0]-1, x[0]+1)
    plt.ylim(x[1]-1, x[1]-1)
    plt.show()

print(img_array[ground_truth[0][0]][ground_truth[0][1]])
ground_truth[0]

