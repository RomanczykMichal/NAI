#s20422 Iris K-NN

#imports
import numpy as np
import pandas as pd
import operator

def read_file(name):
    return pd.read_csv(name, sep=';', names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"])

def devide_dataset_into_trainset(dataset, sizeOfClass, sizeOfSet):
    trainset = []
    for i in range(3):
        trainset.append(dataset[sizeOfClass * i : sizeOfClass * i + sizeOfSet])
    return pd.concat(trainset).reset_index(drop=True)

def devide_dataset_into_testset(dataset, sizeOfClass, sizeOfSet):
    testset = []
    for i in range(3):
        testset.append(dataset[sizeOfClass * i + (sizeOfClass - sizeOfSet) : sizeOfClass * (i + 1)])
    return pd.concat(testset).reset_index(drop=True)

def user_input():
    return input()

def create_vec(dataset):
    vectors = []
    for row in dataset.itertuples():
        vectors.append([row.Sepal_Length, row.Sepal_Width, row.Petal_Length, row.Petal_Width, row.Class])
    return vectors

def calculate_dist(v1, v2):
    return np.sqrt((v2[0]-v1[0])**2 + (v2[1]-v1[1])**2 + (v2[2]-v1[2])**2 + (v2[3]-v1[3])**2) 
 
def calculate_dist_from_train(testVec, trainsetVec):
    distances = []
    for i in range(len(trainsetVec)):
        distances.append([testVec, trainsetVec[i], calculate_dist(testVec, trainsetVec[i])])
    return distances

def find_k_nearest_neigh(testsetVec, trainsetVec, k):
    distances = sorted(calculate_dist_from_train(testsetVec, trainsetVec), key=operator.itemgetter(2))   
    
    print_list(distances, k)

    kNearest = []
    for i in range(k):
        kNearest.append(distances[i])

    return kNearest 
    
def print_list(list, k):
    for i in range(k):
        print(list[i])

        #dokonczyc to zadanie
def make_prediciton(kNearestSet):
    opt = [["Iris-setosa", 0],["Iris-versicolor", 0],["Iris-virginica", 0]]
    for i in range(len(kNearestSet)):
        if (kNearestSet[i][0][4] == "Iris-setosa"):
            opt[0][1] += 1
        elif (kNearestSet[i][0][4] == "Iris-versicolor"):
            opt[1][1] += 1
        elif (kNearestSet[i][0][4] == "Iris-virginica"):
            opt[2][1] += 1

def main():
    dataset = read_file('iris.csv')
    trainset = devide_dataset_into_trainset(dataset, 50, 35)
    testset = devide_dataset_into_testset(dataset, 50, 15)

    print('Podaj k=', end='')
    k = int(user_input())
   
    trainsetVec = create_vec(trainset)
    testsetVec = create_vec(testset)

    #test
    predictions = []
    for i in range(len(testsetVec)):
       kNearest = find_k_nearest_neigh(testsetVec[i], trainsetVec, k)
       predictions.append([testsetVec[i], 'pradykcja algorytmu {}' + str(make_prediciton(kNearest))])
    
    
if __name__ == '__main__':
    main()
