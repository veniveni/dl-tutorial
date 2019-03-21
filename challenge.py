import os, glob, random

import numpy as np
import matplotlib.pylab as plt

from keras.models import Sequential
from keras.layers import Dense
# from keras.utils import np_utils


def load_data(train_dir = r'C:\Work\dl-tutorial\hebrew\train'):
    flist = glob.glob(os.path.join(train_dir,'*.bmp'))
    x = []
    y = []
    for fname in flist:
        x.append(plt.imread(fname))
        y.append(int(os.path.basename(fname)[0]))
    X = np.array(x)
    Y = np.array(y)
    return X,Y

def split_train_test(X, Y, train_frac = 0.85):
    n_samples_per_letter = X.shape[0]/3
    split = int(train_frac*n_samples_per_letter)
    r = range(n_samples_per_letter)
    random.shuffle(r)
    train_indices = r[:split]
    train_indices += map(lambda x: x+n_samples_per_letter, train_indices) + map(lambda x: x + 2*n_samples_per_letter, train_indices)
    test_indices = r[split:]
    test_indices += map(lambda x: x+n_samples_per_letter, test_indices) + map(lambda x: x + 2*n_samples_per_letter, test_indices)
    X_train = X[np.array(train_indices)]
    Y_train = Y[np.array(train_indices)]
    X_test = X[np.array(test_indices)]
    Y_test = Y[np.array(test_indices)]
    return (X_train, Y_train), (X_test, Y_test)

def main():
    X,Y = load_data()
    (X_train, Y_train), (X_test, Y_test) = split_train_test(X,Y)

main()