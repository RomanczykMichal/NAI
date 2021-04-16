import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import sys
from percep import Percepetron

def macierz_omylek(przewidziane, prawdziwe):
    labels = np.asarray(["pl", "en", "fr", "it", "es", "pt"])
    n_labels = labels.size
    label_to_ind = {y: x for x, y in enumerate(labels)}

    przewidziane = np.array([label_to_ind.get(x, n_labels + 1) for x in przewidziane])
    prawdziwe = np.array([label_to_ind.get(x, n_labels + 1) for x in prawdziwe])
    
    sample_weight = np.ones(prawdziwe.shape[0], dtype=np.int64)
    
    ind = np.logical_and(przewidziane < n_labels, prawdziwe < n_labels)
    przewidziane = przewidziane[ind]
    prawdziwe = prawdziwe[ind]
    # also eliminate weights of eliminated items
    sample_weight = sample_weight[ind]

    cm = coo_matrix((sample_weight, (prawdziwe, przewidziane)),
                    shape=(n_labels, n_labels), dtype=np.int64,
                    ).toarray()
    return cm

def calc_precision(macierz, label):
    suma = 0
    for i in range(len(macierz[label])):
        suma += macierz[label][i]
    if suma != 0:
        return macierz[label][label] / suma
    return 0

def calc_recall(macierz, label):
    suma = 0
    for i in range(len(macierz[label])):
        suma += macierz[i][label]   
    if suma != 0:
        return macierz[label][label] / suma
    return 0

def calc_f1_average(macierz):
    f1_avg = 0
    for i in range(6):
        prec = calc_precision(macierz, i)
        reca = calc_recall(macierz, i)
        if (prec + reca) != 0:
            f1_avg += (2 * prec * reca) / (prec + reca)

    return f1_avg / 6


def main():
    trainset = pd.read_csv(sys.argv[1], sep=";")
    testset = pd.read_csv(sys.argv[2], sep=";")
    alpha = float(sys.argv[3])
    k = int(sys.argv[4])

    X_train = [sublist[0:-1] for sublist in trainset.values.tolist()]
    poprawna_train = [sublist[-1] for sublist in trainset.values.tolist()]
    
    X_test = [sublist[0:-1] for sublist in testset.values.tolist()]
    poprawna_test = [sublist[-1] for sublist in testset.values.tolist()]

    p1 = Percepetron(len(X_train[0]), alpha, "pl")
    p2 = Percepetron(len(X_train[0]), alpha, "en")
    p3 = Percepetron(len(X_train[0]), alpha, "fr")
    p4 = Percepetron(len(X_train[0]), alpha, "it")
    p5 = Percepetron(len(X_train[0]), alpha, "es")
    p6 = Percepetron(len(X_train[0]), alpha, "pt")
    
    layer = []
    layer.append(p1)
    layer.append(p2)
    layer.append(p3)
    layer.append(p4)
    layer.append(p5)
    layer.append(p6)

    #uczenie
    for i in range(k):
       for n in range(len(layer)):
           layer[n].train(X_train, poprawna_train)

    #decyzje
    decyzje = []
    for i in range(len(X_test)):
        wstepne_decyzje = []
        for n in range(len(layer)):
            wstepne_decyzje.append([layer[n].decide(X_test[i]), layer[n].klasa, poprawna_test[i]])
        wybor = max(wstepne_decyzje)
        #print(wybor)
        decyzje.append([wybor[1], wybor[2]])

    #stworz macierz
    przewidziane = [item[0] for item in decyzje]
    prawdziwe = [item[1] for item in decyzje]
    macierz = macierz_omylek(przewidziane, prawdziwe)

    #printy
    print(macierz)
    print("F1-miara: " + str(calc_f1_average(macierz)))

    for i in range(len(layer)):
        print("\np" + str(i+1) + " " + str(layer[i].wagi) + "   THETA: " + str(layer[i].theta))

if __name__ == "__main__":
    main()