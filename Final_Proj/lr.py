#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from numpy import exp as exp2
from numpy import infty
from math import sqrt

MAX_ITERS = 10

# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [float(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)


# Computes the activation of a single data point
def activation(featVector, weights, bias):
    a = bias
    for var in range(len(featVector)):
        a += weights[var] * featVector[var]
    return a


# Computes the dot product of two vectors (arrays)
def dotprod(x, w):
    if len(x) != len(w):
        return -1
    prodsum = 0
    for element in range(len(x)):
        prodsum += x[element] * w[element]
    return prodsum


# The gradient descent formula for weights within the weight vector
def gradient_descent(data, w, b):
    w_gradient = [0.0] * len(w)
    b_gradient = 0.0

    for piece in data:
        x = piece[0]
        y = piece[1]
        expo = 1/(1 + exp2(-y * dotprod(x, w) + b))
        for wi in range(len(w)):
            w_gradient[wi] -= y * x[wi] * expo
        b_gradient -= y * expo

    return w_gradient, b_gradient


# The gradient descent formula for bias (excludes the x[xi] term)
def gradient_descent_b(data, w, b):
    b_sum = 0.0
    for piece in data:
        x = piece[0]
        y = piece[1]
        expo_b = exp2(y * dotprod(x, w) + b)
        b_sum -= (1/(1 + expo_b)) * y 
    return b_sum


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numvars = len(data[0][0])
    w = [0.0] * numvars # Forms the weight vector
    b = 0 # Forms the bias

    w_gradient = [0.0] * numvars # Forms the vector to hold the gradient of the weight vector
    b_gradient = infty # Gradient of b
    itr = 0 # Tracks iterations passed

    while itr < MAX_ITERS and sqrt(dotprod(w_gradient, w_gradient) + b_gradient**2) > 0.0001:
        w_gradient, b_gradient = gradient_descent(data, w, b) # Compute the gradient of the weight vector
        for var in range(len(w)): # For each variable in the weight vector
            w_gradient[var] += l2_reg_weight * w[var] * eta
            w[var] = w[var] - (eta * w_gradient[var]) # Update all weights after gradient has been fully computed
        b = b - eta * b_gradient # Also update the bias via gradient descent
        itr += 1
        print("Iteration", itr, "finished.")
    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    # print(activation(w, x, b))
    return activation(w, x, b)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        # print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
