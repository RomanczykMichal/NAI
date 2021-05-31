#K-means
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#obliczanie najblizszego centroidu listy centroidow 
def calc_nearest(point, centroids):
    nearest = 0
    smallest_distance = calc_distance(point[:-1], centroids[0][1])

    for centroid in centroids:
        dist = calc_distance(point[:-1], centroid[1])
        if dist < smallest_distance: 
            nearest = centroid[0]
            smallest_distance = dist

    return nearest

def calc_distance(X, Y):
    dist = 0

    for i in range(len(X)):
        dist += np.power(X[i] - Y[i], 2)

    return dist

def get_points(clusters):
    result = []
    for cluster in clusters:
        result += cluster[1:]
    return result

def plot(centroids, points, axs):
    colors = ['b', 'g', 'r', 'm']

    for i in range(len(centroids)):
        color = colors[i]
        x = centroids[i][1][0]
        y = centroids[i][1][1]
        axs.plot(x, y, 'D' + color)
        
        x_p = [j[0] for j in points if j[-1] == i]
        y_p = [j[1] for j in points if j[-1] == i]
        axs.plot(x_p, y_p, '.' + color)

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
    
    #wykres przed
    fig, axs = plt.subplots(2, figsize=(8,7))
    fig.suptitle('Przed i po algorytmie (tylko 1 i 2 współrzędna)')
    plot(centroids, data_list, axs[0])

    #algorytm K-means
    while zmiana != 0:
        zmiana = 0
        points = get_points(clusters)

        for point in points:
            #oblicz odleglosc punktu od centroidów i wybierz najmniejsza odleglosc.
            nearest = calc_nearest(point, centroids)
            centroid_id = point[-1]

            #Jesli najmniejsza odleglosc jest do innego centroidu to podmien ostatni element punktu na id_centroidu, do ktorego odleglosc jest najmniejsza
            if nearest != centroid_id:
                #Jesli nastapila zmiana usun z poprzedniego klastera punkt i dopisz go do nowego klastera
                clusters[centroid_id].remove(point)    
                point[-1] = nearest
                clusters[nearest].append(point)
                
                if zmiana == 0:
                    zmiana = 1
        
        if zmiana == 1:
            centroids = calculate_centroids(clusters)
        print('Loop done')   
    
    #koncowe wypisywanie
    for cluster in clusters:
        print('Cluster#' + str(cluster[0]) + ':\n', cluster)
    
    #rysowanie wykresow    
    plot(centroids, get_points(clusters), axs[1])
    plt.show()


if __name__ == '__main__':
    main()
