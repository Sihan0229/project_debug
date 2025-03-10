import time
import os
from sklearn import preprocessing
import h5py
from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import AffinityPropagation
import sklearn.metrics as sm
np.set_printoptions(threshold=np.inf)



if __name__=='__main__':
    # print("======GenKeyTime(s)======")
    # GenKeyTime=np.load('./Result/GenKeyTime.npy')
    # print(GenKeyTime)
    # print("\n")

    print("======GenKeyStorage(MB)======")
    GenKeyStorage=np.load('./Result/GenKeyStorage.npy')
    print(GenKeyStorage)
    print("\n")

    print("======GenIndexTime(s)======")
    GenIndexTime=np.load('./Result/GenIndexTime.npy')
    print(GenIndexTime)
    print("\n")

    print("======GenIndexStorage(MB)======")
    GenIndexStorage = np.load('./Result/GenIndexStorage.npy')
    print(GenIndexStorage)
    print("\n")

    print("======TrapdoorTime(ms)======")
    TrapdoorTime = np.load('./Result/GenTrapdoorTime.npy')
    print(TrapdoorTime)
    print("\n")

    print("======Accuracy======")
    Accuracy = np.load('./Result/Accuracy.npy')
    print(Accuracy)
    print("\n")

    print("=====RetrievalTime(ms)======")
    SearchTime = np.load('./Result/SearchTime.npy')
    print(SearchTime)
    print("\n")
