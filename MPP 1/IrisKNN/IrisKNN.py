#s20422 Iris K-NN

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

    kNearest = []
    for i in range(k):
        kNearest.append(distances[i])

    #for i in range(len(kNearest)):
    #   print(distances[i])
   
    return kNearest 

def make_prediciton(kNearestSet):
    opt = [["Iris-setosa", 0],["Iris-versicolor", 0],["Iris-virginica", 0]]
    for i in range(len(kNearestSet)):
        if (kNearestSet[i][1][4] == "Iris-setosa"):
            opt[0][1] += 1
        elif (kNearestSet[i][1][4] == "Iris-versicolor"):
            opt[1][1] += 1
        elif (kNearestSet[i][1][4] == "Iris-virginica"):
            opt[2][1] += 1
    
    return max(opt, key=operator.itemgetter(1))[0] 

def calculate_acc(predictions):
    correct = 0
    for i in range(len(predictions)):
        if predictions[i][0][4] == predictions[i][1]:
            correct += 1
    return (correct/float(len(predictions))) * 100.0

def guess(testsetVec, trainsetVec, k):
    predictions = []
    for i in range(len(testsetVec)):
       kNearest = find_k_nearest_neigh(testsetVec[i], trainsetVec, k)
       predictions.append([testsetVec[i], make_prediciton(kNearest)])
    return predictions

def process_user_input(columnNames):
    wektor = []
    for i in range(len(columnNames) - 1):
        print('Podaj', columnNames[i])
        value = float(user_input())
        wektor.append(value)
    return wektor

def main():
    dataset = read_file('iris.csv')
    trainset = devide_dataset_into_trainset(dataset, 50, 35)
    testset = devide_dataset_into_testset(dataset, 50, 15)

    print('Podaj k=', end='')
    k = int(user_input())
   
    trainsetVec = create_vec(trainset)
    testsetVec = create_vec(testset)

    #test
    predictions = guess(testsetVec, trainsetVec, k)
    for i in range(len(testsetVec)):
       print('TEST#'+str(i+1), predictions[i])
    print("Dokladnosc dla testow dla podanego k ---> " + str(calculate_acc(predictions)))


    #petla do wpisywania kolejnych wartosci
    while(True):
        print('Czy chcesz wpisywac kolejne wektory do sprawdzenia (y/n) ', end='')
        
        wpisywac = user_input()
        if (wpisywac != 'y'):
            break
        
        wektor = process_user_input(dataset.columns)
        predict = make_prediciton(find_k_nearest_neigh(wektor, trainsetVec, k))
        print('Wynik predykcji to', predict)
    
    #na plusa
    print('Dodatkowa wlasnosc.\nGraf efektywnosci w zaleznosci od liczby k')
    n = 105
    y = []
    for i in range(1, n):
        pred = guess(testsetVec, trainsetVec, i)
        y.append(calculate_acc(pred))

    plt.scatter(range(1, n), y, marker='.')
    plt.xlabel('k')
    plt.ylabel('dokladnosc (%)')
    plt.grid()
    plt.show()
        
    
if __name__ == '__main__':
    main()
