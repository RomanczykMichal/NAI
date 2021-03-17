#s20422 Iris K-NN

#imports
import numpy as np
import pandas as pd

def readFile(name):
    return pd.read_csv(name, sep=';', names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"])

def devideDatasetIntoTrain(dataset, sizeOfClass, sizeOfSet):
    trainset = []
    for i in range(3):
        trainset.append(dataset[sizeOfClass * i : sizeOfClass * i + sizeOfSet])
    return pd.concat(trainset).reset_index(drop=True)

def devideDatasetIntoTest(dataset, sizeOfClass, sizeOfSet):
    testset = []
    for i in range(3):
        testset.append(dataset[sizeOfClass * i + (sizeOfClass - sizeOfSet) : sizeOfClass * (i + 1)])
    return pd.concat(testset).reset_index(drop=True)

def userInput():
    return input()

def createVec(dataset):
    vectors = []
    for row in dataset.itertuples():
        vectors.append([row.Sepal_Length, row.Sepal_Width, row.Petal_Length, row.Petal_Width, row.Class])
    return vectors

def calculateDist(v1, v2):
    return np.sqrt((v2[0]-v1[0])**2 + (v2[1]-v1[1])**2 + (v2[2]-v1[2])**2 + (v2[3]-v1[3])**2) 

def main():
    dataset = readFile('iris.csv')
    trainset = devideDatasetIntoTrain(dataset, 50, 35)
    testset = devideDatasetIntoTest(dataset, 50, 15)

    print('Podaj k=', end='')
    k = int(userInput())
   
    trainsetVec = createVec(trainset)
    testsetVec = createVec(testset)

    for i in range(len(testsetVec)):
        for j in range(len(trainsetVec)):
           print('Odleglosc wektora ', testsetVec[i], ' od wektora ', trainsetVec[j], ' jest rowna=', calculateDist(testsetVec[i], trainsetVec[j]))
           



main()
