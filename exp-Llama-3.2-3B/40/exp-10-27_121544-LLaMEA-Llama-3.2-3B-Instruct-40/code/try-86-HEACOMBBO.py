import numpy as np
from scipy.optimize import differential_evolution
import random

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if func.__name__ == 'f1':
            return differential_evolution(func, self.bounds)
        elif func.__name__ == 'f2':
            return differential_evolution(func, self.bounds, x0=self.x0)
        elif func.__name__ == 'f3':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5)
        elif func.__name__ == 'f4':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0)
        elif func.__name__ == 'f5':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx')
        elif func.__name__ == 'f6':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5)
        elif func.__name__ == 'f7':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5)
        elif func.__name__ == 'f8':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5)
        elif func.__name__ == 'f9':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        elif func.__name__ == 'f10':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        elif func.__name__ == 'f11':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        elif func.__name__ == 'f12':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f13':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        elif func.__name__ == 'f14':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f15':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f16':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f17':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f18':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f20':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f21':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f22':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f23':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f24':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def evaluate_fitness(self, individual):
        if self.budget == 0:
            return np.nan, np.nan
        else:
            try:
                result = self(individual)
                return result[0], result[1]
            except Exception as e:
                print(f"Error occurred: {e}")
                return np.nan, np.nan

    def refine_strategy(self, individual):
        if self.budget == 0:
            return individual
        else:
            strategy = random.choice(['sbx', 'rand1'])
            if random.random() < 0.4:
                strategy ='sbx'
            return individual, strategy

    def hybrid_evolution(self, func):
        population_size = 100
        population = [self.x0 + np.random.uniform(-1, 1, self.dim) for _ in range(population_size)]
        for _ in range(100):
            new_population = []
            for individual in population:
                result = self(func, individual)
                if result[0]!= np.nan:
                    new_population.append(result[1])
            population = new_population
        return population[0]

# Usage
def f1(x):
    return np.sum(x**2)

def f2(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4)

def f3(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6)

def f4(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8)

def f5(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10)

def f6(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12)

def f7(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14)

def f8(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16)

def f9(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18)

def f10(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20)

def f11(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22)

def f12(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24)

def f13(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26)

def f14(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28)

def f15(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30)

def f16(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32)

def f17(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32) + 0.000000000001 * np.sum(x**34)

def f18(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32) + 0.000000000001 * np.sum(x**34) + 0.000000000001 * np.sum(x**36)

def f19(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32) + 0.000000000001 * np.sum(x**34) + 0.000000000001 * np.sum(x**36) + 0.000000000001 * np.sum(x**38)

def f20(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32) + 0.000000000001 * np.sum(x**34) + 0.000000000001 * np.sum(x**36) + 0.000000000001 * np.sum(x**38) + 0.000000000001 * np.sum(x**40)

def f21(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32) + 0.000000000001 * np.sum(x**34) + 0.000000000001 * np.sum(x**36) + 0.000000000001 * np.sum(x**38) + 0.000000000001 * np.sum(x**40) + 0.000000000001 * np.sum(x**42)

def f22(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32) + 0.000000000001 * np.sum(x**34) + 0.000000000001 * np.sum(x**36) + 0.000000000001 * np.sum(x**38) + 0.000000000001 * np.sum(x**40) + 0.000000000001 * np.sum(x**42) + 0.000000000001 * np.sum(x**44)

def f23(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32) + 0.000000000001 * np.sum(x**34) + 0.000000000001 * np.sum(x**36) + 0.000000000001 * np.sum(x**38) + 0.000000000001 * np.sum(x**40) + 0.000000000001 * np.sum(x**42) + 0.000000000001 * np.sum(x**44) + 0.000000000001 * np.sum(x**46)

def f24(x):
    return np.sum(x**2) + 0.1 * np.sum(x**4) + 0.01 * np.sum(x**6) + 0.001 * np.sum(x**8) + 0.0001 * np.sum(x**10) + 0.00001 * np.sum(x**12) + 0.000001 * np.sum(x**14) + 0.0000001 * np.sum(x**16) + 0.00000001 * np.sum(x**18) + 0.000000001 * np.sum(x**20) + 0.0000000001 * np.sum(x**22) + 0.00000000001 * np.sum(x**24) + 0.000000000001 * np.sum(x**26) + 0.000000000001 * np.sum(x**28) + 0.000000000001 * np.sum(x**30) + 0.000000000001 * np.sum(x**32) + 0.000000000001 * np.sum(x**34) + 0.000000000001 * np.sum(x**36) + 0.000000000001 * np.sum(x**38) + 0.000000000001 * np.sum(x**40) + 0.000000000001 * np.sum(x**42) + 0.000000000001 * np.sum(x**44) + 0.000000000001 * np.sum(x**46) + 0.000000000001 * np.sum(x**48)

# Usage
def evaluateBBOB(func):
    heacombbo = HEACOMBBO(100, 10)
    best_individual = heacombbo.hybrid_evolution(func)
    return best_individual

# Usage
best_individual = evaluateBBOB(f1)
print(best_individual)