from __future__ import division, print_function
import csv, os, sys
import numpy as np
from SVCSMO import SVCSMO
filepath = os.path.dirname(os.path.abspath(__file__))

def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

def calc_acc(y, y_hat):
    idx = np.where(y_hat == 1)
    TP = np.sum(y_hat[idx] == y[idx])
    idx = np.where(y_hat == -1)
    TN = np.sum(y_hat[idx] == y[idx])
    return float(TP + TN)/len(y)

def calc_mse(y, y_hat):
    return np.nanmean(((y - y_hat) ** 2))

def test_main(filename='data/iris-virginica.txt', C=1.0, kernel_type='linear', epsilon=0.001):
    # Load data
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)
    data = data.astype(float)

    # Split data
    X, y = data[:,0:-1], data[:,-1].astype(int)

    # Initialize model
    model = SVCSMO()

    # Fit model
    support_vectors, iterations = model.fit(X, y)

    # Support vector count
    sv_count = support_vectors.shape[0]

    # Make prediction
    y_hat = model.predict(X)

    # Calculate accuracy
    acc = calc_acc(y, y_hat)
    mse = calc_mse(y, y_hat)

    print("Support vector count: %d" % (sv_count))
    print("bias:\t\t%.3f" % (model.b))
    print("w:\t\t" + str(model.w))
    print("accuracy:\t%.3f" % (acc))
    print("mse:\t%.3f" % (mse))
    print("Converged after %d iterations" % (iterations))

if __name__ == '__main__':
    param = {}
    param['filename'] = './small_data/iris-slwc.txt'
    param['C'] = 0.1
    param['kernel_type'] = 'linear'
    param['epsilon'] = 0.001


    test_main(**param)

