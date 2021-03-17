print('s20422 Iris K-NN');

#imports
import numpy as np
import pandas as pd

#prepare dataset
dataset = pd.read_csv('iris.csv', sep=';' ,names = ["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width", "Class"])


def main():
    #main
    print(dataset)

main()
