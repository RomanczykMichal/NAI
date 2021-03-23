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
        if (np.dot(self.wagi, X) >= self.theta):
            return 1
        return 0

    def train(self, X, poprawna):
        X_vec = X.values.tolist()
        poprawna_vec = poprawna.values.tolist()

        for i in range(len(X_vec)):
            y = self.decide(X_vec[i])
            self.delta(X_vec[i], poprawna[i], y)

    def delta(self, X_vec, decyzja_prawidlowa, faktyczna_decyzja):
        #W` = W + (D - Y)aX
        for i in range(len(self.wagi)):
            self.wagi[i] += float(decyzja_prawidlowa - faktyczna_decyzja) * float(self.alpha) * float(X_vec[i])

        #theta
        self.theta += float(decyzja_prawidlowa - faktyczna_decyzja) * float(self.alpha)
    
def start_test(perceptron, testset):
    testset_vec = testset.values.tolist()
    decisions = []
    for i in range(len(testset)):
        vecX = []
        decisions.append(perceptron.decide(testset_vec[i]))
    return decisions

def calculate_acc(testset_ans, my_ans):
    testset_ans_vec = testset_ans.values.tolist()
    correct = 0

    for i in range(len(my_ans)):
        if (testset_ans_vec[i] == my_ans[i]):
            correct += 1

    return (correct/float(len(my_ans))) * 100.0

def main():
    if (len(sys.argv[1:]) != 3):
        print("Zla liczba argumentow - wymagane 3")
        sys.exit()

    dataset = pd.read_csv(sys.argv[1])
    alpha = sys.argv[2]
    t = sys.argv[3]
    
    #dzielenie na podzbiory
    trainset = dataset.sample(frac = 0.75).reset_index(drop=True)
    testset = dataset.drop(trainset.index).reset_index(drop=True)
    
    perceptron = Perceptron(len(dataset.columns) - 1, alpha)
    accc = []
    for i in range(50):
        perceptron.train(trainset.iloc[:, 0:8], trainset.iloc[:, -1])
        my_dec = start_test(perceptron, testset.iloc[:, 0:8])
        acc = calculate_acc(testset.iloc[:, -1], my_dec)
        print("Test #" + str(i), acc)
        accc.append(acc)

    plt.plot(range(50), accc) 
    plt.show()

    
if __name__ == "__main__":
    main()