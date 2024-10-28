import numpy as np
from scipy.optimize import minimize

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=(dim,))
        self.f_best = np.inf
        self.x_best = None

    def __call__(self, func):
        for _ in range(self.budget):
            # Global Exploration using Differential Evolution
            f_x = func(self.x)
            if f_x < self.f_best:
                self.f_best = f_x
                self.x_best = self.x.copy()
            
            # Local Exploration using Particle Swarm Optimization
            if np.random.rand() < 0.02040816326530612:
                x_p = self.x.copy()
                for i in range(self.dim):
                    if np.random.rand() < 0.5:
                        x_p[i] = self.x[i] + np.random.uniform(-1.0, 1.0)
                    else:
                        x_p[i] = self.x[i] - np.random.uniform(-1.0, 1.0)
                f_x_p = func(x_p)
                if f_x_p < self.f_best:
                    self.f_best = f_x_p
                    self.x_best = x_p.copy()

            # Update the current position
            self.x = (self.x + np.random.uniform(-1.0, 1.0, size=(self.dim,))) / 2

    def get_best_x(self):
        return self.x_best

    def get_best_f(self):
        return self.f_best

# Example usage:
def func(x):
    return np.sum(x**2)

hybrid_DEPSO = HybridDEPSO(budget=100, dim=10)
hybrid_DEPSO()
print(hybrid_DEPSO.get_best_x())
print(hybrid_DEPSO.get_best_f())

# Testing on BBOB test suite of 24 noiseless functions
import numpy as np
from scipy.optimize import minimize
from bbob.testfunctions import BBOB

def hybrid_DEPSO(func, budget, dim):
    hybrid = HybridDEPSO(budget, dim)
    results = []
    for i in range(len(BBOB)):
        x = hybrid.get_best_x()
        f = func(x)
        results.append((x, f))
    return results

# Example usage:
def bbb1(x):
    return x[0]**2 + x[1]**2

def bbb2(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2

def bbb3(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2 + x[21]**2 + x[22]**2 + x[23]**2

def bbb4(x):
    return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 + x[5]**2 + x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 + x[10]**2 + x[11]**2 + x[12]**2 + x[13]**2 + x[14]**2 + x[15]**2 + x[16]**2 + x[17]**2 + x[18]**2 + x[19]**2 + x[20]**2 + x[21]**2 + x[22]**2 + x[23]**2 + x[24]**2

results_bbb1 = hybrid_DEPSO(bbb1, 100, 10)
results_bbb2 = hybrid_DEPSO(bbb2, 100, 10)
results_bbb3 = hybrid_DEPSO(bbb3, 100, 10)
results_bbb4 = hybrid_DEPSO(bbb4, 100, 10)

for i in range(len(results_bbb1)):
    print(f"BBOB Test Suite Function {i+1}:")
    print(f"Function: bbb{i+1}")
    print(f"Best X: {results_bbb1[i][0]}")
    print(f"Best F: {results_bbb1[i][1]}")
    print(f"")