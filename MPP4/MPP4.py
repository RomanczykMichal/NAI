#Naiwny klasyfikator bayesa
import pandas as pd
import numpy as np

#0 - spam
#1 - nie spam

#metoda do wyliczenia ile słow wystepujacych w danym mailu, również znajduje się w zbiorze treningowym
def count_how_many_words(index, train, poprawna, is_spam = 0):
    count = 0
    for n in range(len(train)):
        if poprawna[n] == is_spam:
            count += train[n][index]
    return count

#metoda do policzenia prawdopodobienstwa danego zdarzenia pomijajaca 0.
#Wynika z implementacji tablicy pWar_spam i pWar_nieSpam.
def count_prawdopodobienstwo(py, px):
    prawd = 1
    for n in range(len(px)):
        if px[n] != 0:
            prawd = np.multiply(prawd, px[n])
    return np.multiply(prawd, py)

def main():
    #dane
    words = open('./data/word_list.txt', 'r', encoding = 'utf-8').read().splitlines()
    data = pd.read_csv('./data/data.csv')

    trainset = data.sample(frac = 0.8)
    testset = data.drop(trainset.index)
    
    x_train = [sublist[0:-1] for sublist in trainset.values.tolist()]
    poprawna_train = [sublist[-1] for sublist in trainset.values.tolist()]
    x_test = [sublist[0:-1] for sublist in testset.values.tolist()]
    poprawna_test = [sublist[-1] for sublist in testset.values.tolist()]
    
    #ile jest spamu w zbiorze treningowym
    ile_nieSpam = sum(poprawna_train)
    ile_spam = len(poprawna_train) - ile_nieSpam

    #prawdopodobienstwo
    p_spam = np.divide(ile_spam, len(x_train))
    p_nieSpam = np.subtract(1, p_spam)
    
    #zapisanie odpowiedzi
    anwsers = []

    #Bayes
    for i in range(len(x_test)):
        pWar_spam = np.zeros(len(words))
        pWar_nieSpam = np.zeros(len(words))

        #petla po kazdym slowie
        for w in range(len(words)):
            #ograniczenie tylko do tych slow, ktore wystepuja
            if(x_test[i][w] != 0):
                #obliczam pWar_spam dla danego slowa
                ile_jest_slowa = count_how_many_words(w, x_train, poprawna_train)
                ile_jest_spam = ile_spam
                if ile_jest_slowa == 0: #wygładzanie, do ile_jest_spam dodaje 2 bo tyle jest mozliwosci dla danego slowa
                    ile_jest_slowa += 1
                    ile_jest_spam += 2
                pWar_spam[w] = (np.divide(ile_jest_slowa, ile_jest_spam))

                #obliczam pWar_nieSpam dla danego slowa
                ile_jest_slowa = count_how_many_words(w, x_train, poprawna_train, is_spam = 1)
                ile_jest_nieSpam = ile_nieSpam
                if ile_jest_slowa == 0: #wygładzanie, do ile_jest_spam dodaje 2 bo tyle jest mozliwosci dla danego slowa
                    ile_jest_slowa += 1
                    ile_jest_nieSpam += 2
                pWar_nieSpam[w] = (np.divide(ile_jest_slowa, ile_jest_nieSpam))

        #policzenie wlasciwych prawdopodobienst
        prawdopodobienstwo_spamu = count_prawdopodobienstwo(p_spam, pWar_spam)
        prawdopodobienstwo_nieSpamu = count_prawdopodobienstwo(p_nieSpam, pWar_nieSpam)

        #decyzja
        if prawdopodobienstwo_spamu > prawdopodobienstwo_nieSpamu:
            anwsers.append([0, poprawna_test[i]])
        else:
            anwsers.append([1, poprawna_test[i]])
     

    #macierz omyłek
    #      X  | spam | nspam    <--- zakwalifikowane jako
    #   ------+------+-------
    #   spam  |      |  
    #   ------+------+-------
    #   nspam |      |

    macierz = np.zeros((2,2))
    for i in range(len(anwsers)):
        if anwsers[i][0] == 0 and anwsers[i][1] == 0:
            macierz[0][0] += 1
        elif anwsers[i][0] == 1 and anwsers[i][1] == 0:
            macierz[0][1] += 1
        elif anwsers[i][0] == 0 and anwsers[i][1] == 1:
            macierz[1][0] += 1
        else:
            macierz[1][1] += 1
    print(macierz)

    #miary
    d = (macierz[0][0] + macierz[1][1]) / len(x_test)
    p = macierz[0][0] / (macierz[0, 0] + macierz[1, 0]) 
    r = macierz[0][0] / (macierz[0, 0] + macierz[0, 1]) 
    f = 2 * p * r / (p + r)
    print('dokladnosc: ', d, ' fmiara: ', f)


if __name__ == '__main__':
    main()
