import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.probability = 0.4

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        new_individuals = []
        for _ in range(int(self.budget * self.probability)):
            individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            new_individuals.append(individual)

        for individual in new_individuals:
            func(individual)

        updated_individuals = []
        for individual in new_individuals:
            # Refine the strategy of the individual with a probability of 0.4
            if random.random() < self.probability:
                strategy = random.choice(['sbx', 'rand1', 'log-uniform'])
                if strategy =='sbx':
                    cxpb = random.uniform(0.0, 1.0)
                elif strategy == 'rand1':
                    cxpb = random.uniform(0.0, 1.0)
                else:
                    cxpb = random.uniform(0.0, 1.0)
                mu = random.uniform(0.0, 1.0)
                w = random.uniform(0.0, 1.0)
                if w == 1.0:
                    scaling = random.choice(['uniform', 'log-uniform'])
                else:
                    scaling = 'uniform'
                if scaling == 'log-uniform':
                    scaling = 'uniform'
                updated_individuals.append((individual, cxpb, mu, w, strategy, scaling))
            else:
                updated_individuals.append(individual)

        # Evaluate the fitness of the updated individuals
        fitness_values = []
        for individual in updated_individuals:
            func(individual[0])
            fitness_values.append(individual[1])

        # Select the best individual
        best_individual = min(updated_individuals, key=lambda x: fitness_values[updated_individuals.index(x)])

        return best_individual[0], fitness_values[updated_individuals.index(best_individual)]

# Example usage:
def f1(x):
    return x[0]**2 + x[1]**2

def f2(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1])

def f3(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0])

def f4(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1])

def f5(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0])

def f6(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1])

def f7(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0])

def f8(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1])

def f9(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0])

def f10(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1])

def f11(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0])

def f12(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1])

def f13(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0])

def f14(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1])

def f15(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0])

def f16(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1])

def f17(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1]) + 0.000000001 * x[0]**16 * np.sin(10 * x[0])

def f18(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1]) + 0.000000001 * x[0]**16 * np.sin(10 * x[0]) + 0.000000001 * x[1]**16 * np.cos(10 * x[1])

def f19(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1]) + 0.000000001 * x[0]**16 * np.sin(10 * x[0]) + 0.000000001 * x[1]**16 * np.cos(10 * x[1])

def f20(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1]) + 0.000000001 * x[0]**16 * np.sin(10 * x[0]) + 0.000000001 * x[1]**16 * np.cos(10 * x[1])

def f21(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1]) + 0.000000001 * x[0]**16 * np.sin(10 * x[0]) + 0.000000001 * x[1]**16 * np.cos(10 * x[1])

def f22(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1]) + 0.000000001 * x[0]**16 * np.sin(10 * x[0]) + 0.000000001 * x[1]**16 * np.cos(10 * x[1])

def f23(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1]) + 0.000000001 * x[0]**16 * np.sin(10 * x[0])

def f24(x):
    return x[0]**2 + x[1]**2 + 0.1 * np.sin(10 * x[0]) + 0.1 * np.cos(10 * x[1]) + 0.01 * x[0]**2 * np.sin(10 * x[0]) + 0.01 * x[1]**2 * np.cos(10 * x[1]) + 0.001 * x[0]**4 * np.sin(10 * x[0]) + 0.001 * x[1]**4 * np.cos(10 * x[1]) + 0.0001 * x[0]**6 * np.sin(10 * x[0]) + 0.0001 * x[1]**6 * np.cos(10 * x[1]) + 0.00001 * x[0]**8 * np.sin(10 * x[0]) + 0.00001 * x[1]**8 * np.cos(10 * x[1]) + 0.000001 * x[0]**10 * np.sin(10 * x[0]) + 0.000001 * x[1]**10 * np.cos(10 * x[1]) + 0.0000001 * x[0]**12 * np.sin(10 * x[0]) + 0.0000001 * x[1]**12 * np.cos(10 * x[1]) + 0.00000001 * x[0]**14 * np.sin(10 * x[0]) + 0.00000001 * x[1]**14 * np.cos(10 * x[1]) + 0.000000001 * x[0]**16 * np.sin(10 * x[0])

# Example usage:
def main():
    algorithm = HybridEvolutionaryAlgorithm(100, 2)
    def func(x):
        return x[0]**2 + x[1]**2
    best_individual, _ = algorithm(func)
    print(best_individual)

if __name__ == "__main__":
    main()