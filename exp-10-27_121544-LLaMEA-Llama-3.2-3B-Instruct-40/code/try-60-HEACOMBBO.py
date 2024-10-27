import numpy as np
import random

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.fitness = np.zeros(self.dim)
        self.logger = []

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if func.__name__ == 'f1':
            return self.differential_evolution(func, self.bounds)
        elif func.__name__ == 'f2':
            return self.differential_evolution(func, self.bounds, x0=self.x0)
        elif func.__name__ == 'f3':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5)
        elif func.__name__ == 'f4':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0)
        elif func.__name__ == 'f5':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx')
        elif func.__name__ == 'f6':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5)
        elif func.__name__ == 'f7':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5)
        elif func.__name__ == 'f8':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5)
        elif func.__name__ == 'f9':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        elif func.__name__ == 'f10':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        elif func.__name__ == 'f11':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        elif func.__name__ == 'f12':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f13':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        elif func.__name__ == 'f14':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f15':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f16':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f17':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f18':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f20':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f21':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f22':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f23':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f24':
            return self.differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def differential_evolution(self, func, bounds, x0=None, tol=1e-5, x0_init=None, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform'):
        if x0 is None:
            x0 = np.random.uniform(bounds[0][0], bounds[0][1], self.dim)
        if x0_init is not None:
            x0 = x0_init
        if strategy =='sbx':
            cxpb = cxpb
        if strategy == 'rand1':
            cxpb = cxpb
        if scaling == 'uniform':
            w = w
        if scaling == 'log-uniform':
            w = w

        def f(x):
            return func(x)

        def fitness(x):
            return f(x)

        def mutate(x):
            if random.random() < cxpb:
                idx = random.randint(0, self.dim-1)
                x[idx] = x[idx] + random.uniform(-1, 1)
                if x[idx] < bounds[idx][0]:
                    x[idx] = bounds[idx][0]
                elif x[idx] > bounds[idx][1]:
                    x[idx] = bounds[idx][1]
            return x

        def crossover(x1, x2):
            idx = random.randint(0, self.dim-1)
            return x1[:idx] + x2[idx:]

        def selection(x):
            fitness_x = [fitness(x) for x in x]
            idx = np.argsort(fitness_x)
            return x[idx[0]]

        def mutate_x(x):
            return mutate(x)

        def crossover_x(x1, x2):
            return crossover(x1, x2)

        def selection_x(x):
            return selection(x)

        def initialize_population(pop_size):
            population = []
            for _ in range(pop_size):
                individual = mutate_x(np.random.uniform(bounds[0][0], bounds[0][1], self.dim))
                population.append(individual)
            return population

        def update_population(population):
            new_population = []
            for _ in range(pop_size):
                parent1 = selection_x(population)
                parent2 = selection_x(population)
                child = crossover_x(parent1, parent2)
                new_population.append(child)
            return new_population

        population = initialize_population(pop_size)
        for _ in range(100):
            population = update_population(population)
            self.logger.append([fitness(x) for x in population])

        best_individual = min(population, key=fitness)
        return best_individual, np.min([fitness(x) for x in population])

# Usage
def f1(x):
    return sum(x**2)

def f2(x):
    return sum(x**2) + 1

def f3(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x)

def f4(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x)

def f5(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x)

def f6(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x)

def f7(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x)

def f8(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f9(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f10(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f11(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f12(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f13(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f14(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f15(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f16(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f17(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f18(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f19(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f20(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f21(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f22(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f23(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

def f24(x):
    return sum(x**2) + 1 + np.sin(2*np.pi*x) + np.cos(2*np.pi*x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x) + np.exp(x)

budget = 100
dim = 10
optimizer = HEACOMBBO(budget, dim)

for i in range(24):
    func = globals()[f'f{i+1}']
    best_individual, best_fitness = optimizer(func)
    print(f'Function: f{i+1}, Best Individual: {best_individual}, Best Fitness: {best_fitness}')