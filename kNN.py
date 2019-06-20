import random
import csv
import math
import operator
import numpy as np
def loadData(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset = list(lines)
        dataset = dataset[1:250]
        for x in range(len(dataset)-1):
            for y in range(1,4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x][1:5])
            else:
                testSet.append(dataset[x][1:5])
                
def distanceCalc(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance = pow((instance1[x] - instance2[x]),2)
        return(math.sqrt(distance))
        
def getN(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = distanceCalc(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
# =============================================================================
#     distan = np.zeros([len(distances),1])
#     for i in range(0,len(distances)):
#         dis = distances[i][0][3]
#         if float(dis) == 0:
#             distan[i] = distances[i][1]*4
# =============================================================================
    distances.sort(key=operator.itemgetter(1))     
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortVotes = sorted(classVotes, key=classVotes.__getitem__, reverse=True)
    return(sortVotes[0])        


# =============================================================================
# neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
# response = getResponse(neighbors)
# print(response)
# =============================================================================
def getAccur(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct = correct + 1
    return((correct/float(len(testSet)))*100)

def main():
    trainingSet=[]
    testSet=[]
    split = 0.67
    loadData('data_g_color_rgb.csv', split, trainingSet, testSet)
    predictions=[]
    k = 5
    for x in range(len(testSet)):
        neighbors = getN(trainingSet, testSet[x],k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccur(testSet, predictions)
    print(accuracy)

main()