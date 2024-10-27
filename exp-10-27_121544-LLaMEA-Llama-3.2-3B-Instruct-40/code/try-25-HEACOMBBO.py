import numpy as np
from scipy.optimize import differential_evolution

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.probability = 0.4

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
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f17':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f18':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
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

    def refine_strategy(self, individual):
        strategy = np.random.choice(['sbx', 'rand1', 'uniform', 'log-uniform'], p=[0.4, 0.3, 0.2, 0.1])
        if strategy =='sbx':
            individual['cxpb'] = np.random.uniform(0.0, 1.0)
            individual['mu'] = np.random.uniform(0.0, 1.0)
            individual['w'] = np.random.uniform(0.0, 1.0)
        elif strategy == 'rand1':
            individual['cxpb'] = np.random.uniform(0.0, 1.0)
            individual['mu'] = np.random.uniform(0.0, 1.0)
            individual['w'] = np.random.uniform(0.0, 1.0)
        elif strategy == 'uniform':
            individual['cxpb'] = np.random.uniform(0.0, 1.0)
            individual['mu'] = np.random.uniform(0.0, 1.0)
            individual['w'] = np.random.uniform(0.0, 1.0)
        elif strategy == 'log-uniform':
            individual['cxpb'] = np.random.uniform(0.0, 1.0)
            individual['mu'] = np.random.uniform(0.0, 1.0)
            individual['w'] = np.random.uniform(0.0, 1.0)
        return individual

    def evaluate_fitness(self, individual):
        if individual['strategy'] =='sbx':
            individual['func'] = lambda x: individual['f'](x, self.budget, self.dim, strategy='sbx', cxpb=individual['cxpb'], mu=individual['mu'], w=individual['w'])
        elif individual['strategy'] == 'rand1':
            individual['func'] = lambda x: individual['f'](x, self.budget, self.dim, strategy='rand1', cxpb=individual['cxpb'], mu=individual['mu'], w=individual['w'])
        elif individual['strategy'] == 'uniform':
            individual['func'] = lambda x: individual['f'](x, self.budget, self.dim, strategy='uniform', cxpb=individual['cxpb'], mu=individual['mu'], w=individual['w'])
        elif individual['strategy'] == 'log-uniform':
            individual['func'] = lambda x: individual['f'](x, self.budget, self.dim, strategy='log-uniform', cxpb=individual['cxpb'], mu=individual['mu'], w=individual['w'])
        return individual['func']()

# Usage
def f1(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**2)

def f2(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**3)

def f3(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**4)

def f4(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**5)

def f5(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**6)

def f6(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**7)

def f7(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**8)

def f8(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**9)

def f9(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**10)

def f10(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**11)

def f11(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**12)

def f12(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**13)

def f13(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**14)

def f14(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**15)

def f15(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**16)

def f16(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**17)

def f17(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**18)

def f18(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**19)

def f19(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**20)

def f20(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**21)

def f21(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**22)

def f22(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**23)

def f23(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**24)

def f24(x, budget, dim, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5):
    return sum(x**25)

def evaluateBBOB(func, budget, dim):
    heacombbo = HEACOMBBO(budget, dim)
    best_individual = None
    best_fitness = float('inf')
    for _ in range(100):
        individual = heacombbo.x0
        for _ in range(heacombbo.budget):
            individual = heacombbo.refine_strategy(individual)
            individual['func'] = func(individual)
            fitness = individual['func']()
            if fitness < best_fitness:
                best_individual = individual
                best_fitness = fitness
    return best_individual, best_fitness

# Usage
best_individual, best_fitness = evaluateBBOB(f1, 10, 2)
print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)