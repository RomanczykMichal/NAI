#K-means
import pandas as pd
import numpy as np
import sys
import random as ran

#zbior danych to iris.csv z poprzednich projektow

def calculate_centroids(clusters):
    centroids = []

    for n in range(len(clusters)):
        centroids.append([n])
        points = clusters[n][1:]
        wspolrzedne_centroidu = [sum(i)/len(points) for i in zip(*points)] #ten zip jest geeksforgeeks.org
        centroids[n].append(wspolrzedne_centroidu[:-1]) #do przedostatniego, poniewaz kazdy punkt zawierał też numer klastra, w ktorym sie znajduje

    return centroids

def main():
    #inicjalizacja
    if (len(sys.argv[1:]) != 1):
        print("podaj w argumencie parametr k")
        sys.exit()

    K = (int) (sys.argv[1])
    data = pd.read_csv('.\data\iris.csv')
    
    #przypisanie do losowych klastrow
    clusters = []
    for n in range(K):
        clusters.append([n])

    data_list = data.values.tolist()
    ran.shuffle(data_list)
    clust_num = 0

    for n in range(len(data_list)):
        #zapisz w punkcie do ktorego klastra nalezy, dodaj do listy klastrow odpowiedni punkt
        clust_id = clust_num % K
        data_list[n].append(clust_id)
        clusters[clust_id].append(data_list[n])
        clust_num += 1

    #1-nastapila jakas zmiana, 0-nie nastapila zadna zmiana
    zmiana = 1
    centroids = calculate_centroids(clusters)

    #algorytm K-means
    while zmiana != 0:
        zmiana = 0

        #for point in range(len(data_list)):
        #oblicz odleglosc punktu od centroidów i wybierz najmniejsza odleglosc.
        #Jesli najmniejsza odleglosc jest do innego centroidu to podmien ostatni element punktu na id_centroidu, do ktorego odleglosc jest najmniejsza
        #Jesli nastapila zmiana usun z poprzedniego klastera punkt i dopisz go do nowego klastera
        #Jesli nastapila zmiana -> zmiana = 1

    

    print(clusters)

if __name__ == '__main__':
    main()
