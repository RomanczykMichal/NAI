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
        #warunek i decyzja algorytmu
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
        self.wagi = normalize_vec(self.wagi)
        
        #theta
        self.theta += (decyzja_prawidlowa - faktyczna_decyzja) * self.alpha
        dlugosc_wag = np.sqrt(calc_dot(self.wagi, self.wagi))
        self.theta /= dlugosc_wag
    
def start_test(perceptron, testset):
    #metoda, ktora jest odpowiedzialna za przeprowadzenie testu
    decisions = []
    for i in range(len(testset)):
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

def normalize_vec(X):
    X_norm = []
    dlugosc = np.sqrt(calc_dot(X, X))
    for i in range(len(X)):
        X_norm.append(X[i] / dlugosc)
    return X_norm

def main():
    if (len(sys.argv[1:]) != 3):
        print("Zla liczba argumentow - wymagane 3")
        sys.exit()

    data = pd.read_csv(sys.argv[1])
    alpha = float(sys.argv[2])
    t = float(sys.argv[3])
    
    #dzielenie na podzbiory
    datastand = data.values.tolist()
    dataset = pd.DataFrame(datastand)

    trainset = dataset.sample(frac = 0.75)
    testset = dataset.drop(trainset.index)
    
    X_train = [sublist[0:-1] for sublist in trainset.values.tolist()]
    poprawna_train = [sublist[-1] for sublist in trainset.values.tolist()]
    
    X_test = [sublist[0:-1] for sublist in testset.values.tolist()]
    poprawna_test = [sublist[-1] for sublist in testset.values.tolist()]
    
    perceptron = Perceptron(len(dataset.columns) - 1, alpha)
    
    #uczenie
    n = 1
    accset = [0]

    while(accset[-1] < t):
        perceptron.train(X_train, poprawna_train)
        my_dec = start_test(perceptron, X_test)
        acc = calculate_acc(poprawna_test, my_dec)
        accset.append(acc)
        print("Test #" + str(n), "Dokladnosc " + str(acc), "\nWagi", perceptron.wagi, "Prog " + str(perceptron.theta))
        n += 1

    print("Perceptron wyuczony na odpowiedni próg.\nWylacz graf zeby kontynuowac")
    plt.plot(range(n), accset)
    plt.xlabel("liczba szkoleń")
    plt.ylabel("skuteczność")
    plt.show()

    #glowna petla
    while (True):
        print("Wybierz opcje\n1. Podaj wektor do klasyfikacji.\n2. Wykonaj dodatkowy krok uczenia na podstawie podanego wektora.\n3. Pokaz wage i prog perceptronu.")

        opt = input()

        if (opt == '1'):
            vec = []
            for i in range(len(X_train[0])):
                print("Podaj " + str(i) + " wspolrzedna:", end=" ")
                vec.append(float(input()))
            decision = perceptron.decide(vec)
            print("Decyzja: " + str(decision))
        elif(opt == '2'):
            vec = []
            for i in range(len(X_train[0])):
                print("Podaj " + str(i) + " wspolrzedna:", end=" ")
                vec.append(float(input()))
            print("Podaj poprawny wynik dla tego przypadku.", end=" ")
            poprawna = float(input())
            perceptron.train([vec], [poprawna])
            print("Uczenie udane.")
        elif(opt == '3'):
            print("Wagi", perceptron.wagi, "Prog", perceptron.theta)
        else:
            break
    
if __name__ == "__main__":
    main()

#*Wnioski
#-Zrezygnowalem ze standaryzacji, poniewaz zaburzala ona wartosci danych
#Dodatkowo przysparzala ona problemow zwiazanych z przyjmowaniem wartosci od uzytkownika,
#poniewaz podany wektor musial byc dodatkowo standaryzowany z pozostalymi danymi 
#-Wazna jest dobre dobranie alphy, aby perceptron mial szanse dokladnie ustawic swoje progi.
#--Dla zbioru iris Przy alphie=0.5 otrzymamy wynik w 1 iteracji, ale progi beda zupenie inne, niz przy alphie=0.01 i 20 iteracjach.
#  Druga opcja ma wieksza skutecznosc odpowiedzi przy podawaniu wektora recznie.
#--Dla zbioru sonar, ktory ma o wiele wieksza liczbe kolumn uczenie na bardzo malej alphie trwa zdecydowanie zbyt dlugo i wartosc z
#  przedzialu 0.01-0.03 jest optymalna
