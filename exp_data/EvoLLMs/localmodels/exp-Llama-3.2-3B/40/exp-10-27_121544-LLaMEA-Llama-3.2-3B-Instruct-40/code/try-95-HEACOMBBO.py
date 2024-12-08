import numpy as np
import random
from scipy.optimize import differential_evolution

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population_size = 20
        self.population = [np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        for _ in range(self.budget):
            fitness = []
            for individual in self.population:
                # Refine the strategy with probability 0.4
                if random.random() < 0.4:
                    strategy = random.choice(['sbx', 'rand1', 'log-uniform'])
                else:
                    strategy = random.choice(['sbx', 'rand1', 'log-uniform', 'uniform'])
                fitness.append(func(individual, strategy=strategy))

            # Evaluate the best individual
            self.population = [min(fitness)]
            self.population = [min(fitness) if fitness else np.nan for fitness in fitness]

        # Return the best individual
        best_individual = min(self.population)
        return best_individual, np.nan

# Example usage:
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1])

def f3(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1])

def f4(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1])

def f5(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1])

def f6(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1])

def f7(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1])

def f8(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1])

def f9(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1])

def f10(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1])

def f11(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1])

def f12(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1])

def f13(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f14(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f15(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f16(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f17(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f18(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f19(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f20(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f21(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f22(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f23(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def f24(x):
    return x[0]**2 + x[1]**2 + 0.1*np.sin(10*x[0]) + 0.1*np.cos(10*x[1]) + 0.01*np.sin(100*x[0]) + 0.01*np.cos(100*x[1]) + 0.001*np.sin(1000*x[0]) + 0.001*np.cos(1000*x[1]) + 0.0001*np.sin(10000*x[0]) + 0.0001*np.cos(10000*x[1]) + 0.00001*np.sin(100000*x[0]) + 0.00001*np.cos(100000*x[1]) + 0.000001*np.sin(1000000*x[0]) + 0.000001*np.cos(1000000*x[1]) + 0.0000001*np.sin(10000000*x[0]) + 0.0000001*np.cos(10000000*x[1]) + 0.00000001*np.sin(100000000*x[0]) + 0.00000001*np.cos(100000000*x[1]) + 0.000000001*np.sin(1000000000*x[0]) + 0.000000001*np.cos(1000000000*x[1]) + 0.0000000001*np.sin(10000000000*x[0]) + 0.0000000001*np.cos(10000000000*x[1]) + 0.00000000001*np.sin(100000000000*x[0]) + 0.00000000001*np.cos(100000000000*x[1]) + 0.000000000001*np.sin(1000000000000*x[0]) + 0.000000000001*np.cos(1000000000000*x[1])

def evaluateBBOB(func, budget, dim):
    heacombbo = HEACOMBBO(budget, dim)
    for i in range(budget):
        best_individual, _ = heacombbo()
        if i % 10 == 0:
            print(f"Iteration {i}: Best individual = {best_individual}")

    return best_individual

evaluateBBOB(f1, 50, 2)
evaluateBBOB(f2, 50, 2)
evaluateBBOB(f3, 50, 2)
evaluateBBOB(f4, 50, 2)
evaluateBBOB(f5, 50, 2)
evaluateBBOB(f6, 50, 2)
evaluateBBOB(f7, 50, 2)
evaluateBBOB(f8, 50, 2)
evaluateBBOB(f9, 50, 2)
evaluateBBOB(f10, 50, 2)
evaluateBBOB(f11, 50, 2)
evaluateBBOB(f12, 50, 2)
evaluateBBOB(f13, 50, 2)
evaluateBBOB(f14, 50, 2)
evaluateBBOB(f15, 50, 2)
evaluateBBOB(f16, 50, 2)
evaluateBBOB(f17, 50, 2)
evaluateBBOB(f18, 50, 2)
evaluateBBOB(f19, 50, 2)
evaluateBBOB(f20, 50, 2)
evaluateBBOB(f21, 50, 2)
evaluateBBOB(f22, 50, 2)
evaluateBBOB(f23, 50, 2)
evaluateBBOB(f24, 50, 2)