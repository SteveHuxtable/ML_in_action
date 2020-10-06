from math import exp
from numpy import mat, shape
import numpy as np

def load_dataset():
    data_mat = []; label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat

data_mat, label_mat = load_dataset()
test_data_mat = mat(data_mat)
test_label_mat = mat(label_mat).transpose()
test_weight = np.ones((3, 1))
print(np.array(data_mat))
print(test_data_mat * test_weight)

def sigmoid(z):
    return 1.0/(1+exp(-z))

def grad_ascent(data_mat, label_mat):
    data_mat = mat(data_mat)
    label_mat = mat(label_mat).transpose()
    m, n = shape(data_mat)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n, 1)) # the starting point
    for k in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = label_mat - h
        weights = weights + alpha * data_mat.transpose() * error
    return weights

