import numpy as np

class Percepetron(object):
    theta = 1
    
    def __init__(self, width, alpha, klasa):
        self.wagi = np.ones(width)
        self.alpha = alpha
        self.klasa = klasa
    
    def decide(self, X):
        net = self.calc_net(X)
        return 2 / (1 + np.exp(-net)) - 1

    def calc_net(self, X):
        return calc_dot(self.wagi, X) - self.theta

    def train(self, X, poprawna):
        for i in range(len(X)):
            y = self.decide(X[i])
            self.delta(X[i], poprawna[i], y)

    def delta(self, X_vec, decyzja_prawidlowa, faktyczna_decyzja):
        przewidywana = -1
        if decyzja_prawidlowa == self.klasa:
            przewidywana = 1

        #W` = W + (D - Y)aX
        for i in range(len(self.wagi)):
            self.wagi[i] += (przewidywana - faktyczna_decyzja) * self.alpha * X_vec[i]
        dlugosc_wag = np.sqrt(calc_dot(self.wagi, self.wagi))
        self.wagi = normalize_vec(self.wagi)
        
        #theta
        self.theta -= (przewidywana - faktyczna_decyzja) * self.alpha
        self.theta /= dlugosc_wag
        self.theta *= 50
        
        
def calc_dot(W, X):
    sum = 0
    for i in range(len(W)):
        sum += W[i] * X[i]
    return sum

def normalize_vec(X):
    X_norm = []
    dlugosc = np.sqrt(calc_dot(X, X))
    for i in range(len(X)):
        X_norm.append(50 * X[i] / dlugosc)
    return X_norm
