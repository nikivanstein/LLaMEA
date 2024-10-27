import numpy as np

class DEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.v = np.random.uniform(-1.0, 1.0, (budget, dim))
        self.f_best = np.inf
        self.x_best = None
        self.mut_prob = 0.2

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the objective function
            f = func(self.x[i])
            
            # Update the personal best
            if f < self.f_best:
                self.f_best = f
                self.x_best = self.x[i]
                
            # Update the global best
            if f < func(self.x_best):
                self.f_best = f
                self.x_best = self.x[i]
                
            # Update the velocity
            self.v[i] = 0.5 * (self.v[i] + 0.5 * np.random.uniform(-1.0, 1.0, (self.dim,)))
            self.v[i] = self.v[i] + 1.0 * np.random.uniform(-1.0, 1.0, (self.dim,)) * (self.x[i] - self.x_best)
            self.v[i] = self.v[i] + 0.5 * np.random.uniform(-1.0, 1.0, (self.dim,)) * (self.x[i] - self.x[i])
            
            # Update the position
            self.x[i] = self.x[i] + self.v[i]
            
            # Adaptive mutation probability
            if np.random.rand() < self.mut_prob:
                # Randomly select a dimension to mutate
                idx = np.random.choice(self.dim)
                # Mutate the selected dimension
                self.x[i, idx] += np.random.uniform(-1.0, 1.0)

# Test the algorithm
def func(x):
    return np.sum(x**2)

budget = 100
dim = 10
optimizer = DEPSO(budget, dim)
for x in optimizer(func):
    print(func(x))

# BBOB test suite
import numpy as np
from scipy.special import comb
from functools import reduce
from operator import add
from itertools import product
import random

def bbb_1d(x):
    return np.sum(x**2)

def bbb_2d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2)

def bbb_3d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2)

def bbb_4d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2)

def bbb_5d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2)

def bbb_6d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2)

def bbb_7d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2)

def bbb_8d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2)

def bbb_9d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2)

def bbb_10d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2)

def bbb_11d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2)

def bbb_12d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2)

def bbb_13d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2)

def bbb_14d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2)

def bbb_15d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2)

def bbb_16d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2)

def bbb_17d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2) + np.sum(x[:, 16]**2)

def bbb_18d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2) + np.sum(x[:, 16]**2) + np.sum(x[:, 17]**2)

def bbb_19d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2) + np.sum(x[:, 16]**2) + np.sum(x[:, 17]**2) + np.sum(x[:, 18]**2)

def bbb_20d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2) + np.sum(x[:, 16]**2) + np.sum(x[:, 17]**2) + np.sum(x[:, 18]**2) + np.sum(x[:, 19]**2)

def bbb_21d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2) + np.sum(x[:, 16]**2) + np.sum(x[:, 17]**2) + np.sum(x[:, 18]**2) + np.sum(x[:, 19]**2) + np.sum(x[:, 20]**2)

def bbb_22d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2) + np.sum(x[:, 16]**2) + np.sum(x[:, 17]**2) + np.sum(x[:, 18]**2) + np.sum(x[:, 19]**2) + np.sum(x[:, 20]**2) + np.sum(x[:, 21]**2)

def bbb_23d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2) + np.sum(x[:, 16]**2) + np.sum(x[:, 17]**2) + np.sum(x[:, 18]**2) + np.sum(x[:, 19]**2) + np.sum(x[:, 20]**2) + np.sum(x[:, 21]**2) + np.sum(x[:, 22]**2)

def bbb_24d(x):
    return np.sum(x[:, 0]**2) + np.sum(x[:, 1]**2) + np.sum(x[:, 2]**2) + np.sum(x[:, 3]**2) + np.sum(x[:, 4]**2) + np.sum(x[:, 5]**2) + np.sum(x[:, 6]**2) + np.sum(x[:, 7]**2) + np.sum(x[:, 8]**2) + np.sum(x[:, 9]**2) + np.sum(x[:, 10]**2) + np.sum(x[:, 11]**2) + np.sum(x[:, 12]**2) + np.sum(x[:, 13]**2) + np.sum(x[:, 14]**2) + np.sum(x[:, 15]**2) + np.sum(x[:, 16]**2) + np.sum(x[:, 17]**2) + np.sum(x[:, 18]**2) + np.sum(x[:, 19]**2) + np.sum(x[:, 20]**2) + np.sum(x[:, 21]**2) + np.sum(x[:, 22]**2) + np.sum(x[:, 23]**2)

# BBOB test suite results
results = {
    "bbb_1d": bbb_1d,
    "bbb_2d": bbb_2d,
    "bbb_3d": bbb_3d,
    "bbb_4d": bbb_4d,
    "bbb_5d": bbb_5d,
    "bbb_6d": bbb_6d,
    "bbb_7d": bbb_7d,
    "bbb_8d": bbb_8d,
    "bbb_9d": bbb_9d,
    "bbb_10d": bbb_10d,
    "bbb_11d": bbb_11d,
    "bbb_12d": bbb_12d,
    "bbb_13d": bbb_13d,
    "bbb_14d": bbb_14d,
    "bbb_15d": bbb_15d,
    "bbb_16d": bbb_16d,
    "bbb_17d": bbb_17d,
    "bbb_18d": bbb_18d,
    "bbb_19d": bbb_19d,
    "bbb_20d": bbb_20d,
    "bbb_21d": bbb_21d,
    "bbb_22d": bbb_22d,
    "bbb_23d": bbb_23d,
    "bbb_24d": bbb_24d,
}

# Test the algorithm
budget = 100
dim = 10
optimizer = DEPSO(budget, dim)
for f in results.values():
    print(f"Function: {f.__name__}")
    for x in optimizer(f):
        print(f"Value: {f(x)}")
    print()