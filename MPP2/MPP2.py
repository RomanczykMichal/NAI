import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

class Perceptron:
    theta = 1
    
    def __init__(self, width, alpha):
        self.wagi = np.ones(width)
        self.alpha = alpha
    
    def decide(self, X):
        if (calc_dot(self.wagi, X) >= self.theta):
            return 1
        return 0

    def train(self, X, poprawna):
        for i in range(len(X)):
            y = self.decide(X[i])
            self.delta(X[i], poprawna[i], y)

    def delta(self, X_vec, decyzja_prawidlowa, faktyczna_decyzja):
        #W` = W + (D - Y)aX
        for i in range(len(self.wagi)):
            self.wagi[i] += (decyzja_prawidlowa - faktyczna_decyzja) * self.alpha * X_vec[i]

        #theta
        self.theta += (decyzja_prawidlowa - faktyczna_decyzja) * self.alpha
    
def start_test(perceptron, testset):
    decisions = []
    for i in range(len(testset)):
        vecX = []
        decisions.append(perceptron.decide(testset[i]))
    return decisions

def calculate_acc(testset_ans, my_ans):
    correct = 0

    for i in range(len(my_ans)):
        if (testset_ans[i] == my_ans[i]):
            correct += 1

    return (correct / float(len(my_ans))) * 100.0

def calc_dot(W, X):
    sum = 0
    for i in range(len(W)):
        sum += W[i] * X[i]
    return sum

def standarize(X):
    #formula ze strony
    #https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/
    X_res = X

    count = len(X)
    for i in range(len(X[0])):
        sum = 0
        for j in range(len(X)):
            sum += X[j][i]
        mean = sum / count

        sum_sd = 0 
        for j in range(len(X)):
            sum_sd += (X[j][i] - mean) ** 2
        standard_deviation = np.sqrt(sum_sd / count)

        for j in range(len(X)):
            X_res[j][i] = ((X[j][i] - mean) / standard_deviation)

    return X_res


def main():
    if (len(sys.argv[1:]) != 3):
        print("Zla liczba argumentow - wymagane 3")
        sys.exit()

    dataset = pd.read_csv(sys.argv[1])
    alpha = float(sys.argv[2])
    t = float(sys.argv[3])
    
    #dzielenie na podzbiory
    trainset = dataset.sample(frac = 0.75).reset_index(drop=True)
    testset = dataset.drop(trainset.index).reset_index(drop=True)
    
    X_train = standarize([sublist[0:-1] for sublist in trainset.values.tolist()])
    poprawna_train = [sublist[-1] for sublist in trainset.values.tolist()]
    
    X_test = standarize([sublist[0:-1] for sublist in testset.values.tolist()])
    poprawna_test = [sublist[-1] for sublist in testset.values.tolist()]
    
    perceptron = Perceptron(len(dataset.columns) - 1, alpha)

    #petla
    n = 100
    accset = []
    for i in range(n):
        perceptron.train(X_train, poprawna_train)
        my_dec = start_test(perceptron, X_test)
        acc = calculate_acc(poprawna_test, my_dec)
        print("Test #" + str(i), acc)
        accset.append(acc)

    plt.plot(range(n), accset) 
    plt.show()

    
if __name__ == "__main__":
    main()