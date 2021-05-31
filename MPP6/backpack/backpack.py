import numpy as np

def brute_force(v, w, res, n, objetosc):
    #w - wagi
    #v - wartosci
    #res - wektor charakterystyczny
    #n - liczba obietkow
    result = []
    best_value = 0
    best_wagi = 0
    
    for i in range(2**n):
        j = n - 1
        temp_waga = 0
        temp_value = 0
        
        while (res[j] != 0 and j>=0):
            res[j] = 0
            j -= 1
        res[j] = 1

        for k in range(n):
            if res[k] == 1:
                temp_waga += w[k]
                temp_value += v[k]

        if temp_value > best_value and temp_waga <= objetosc:
            best_value = temp_value
            result = list(res)

    return result

def main():
    print("Podaj objetosc plecaka")
    objetosc = (int)(input())
    print("Podaj liczbe przedmiotow")
    n = (int)(input())
    
    przedmioty = []

    for i in range(n):
        print("Podaj dane przedmiotu " + str(i+1))
        przedmiot = str(input()).split(' ')
        przedmiot = list(map(int, przedmiot))
        przedmioty.append(przedmiot)

    A = np.zeros(n)
    result = brute_force([x[0] for x in przedmioty], [x[1] for x in przedmioty], A, n, objetosc) 

    odp = [przedmioty[i] for i in range(len(result)) if result[i] == 1]
    
    print('\nOdpowiedz:')
    for i in range(len(odp)):
        print(odp[i][0], odp[i][1])
    #print(result)
    return

if __name__ == "__main__":
    main()