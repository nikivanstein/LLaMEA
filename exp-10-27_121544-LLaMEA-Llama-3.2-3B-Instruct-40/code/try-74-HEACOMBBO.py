import numpy as np
from scipy.optimize import differential_evolution
import random

class HEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.fitness_history = np.zeros(budget)

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

    def evaluate_fitness(self, func, individual):
        if func.__name__ == 'f1':
            return -func(individual)
        elif func.__name__ == 'f2':
            return -func(individual)
        elif func.__name__ == 'f3':
            return -func(individual)
        elif func.__name__ == 'f4':
            return -func(individual)
        elif func.__name__ == 'f5':
            return -func(individual)
        elif func.__name__ == 'f6':
            return -func(individual)
        elif func.__name__ == 'f7':
            return -func(individual)
        elif func.__name__ == 'f8':
            return -func(individual)
        elif func.__name__ == 'f9':
            return -func(individual)
        elif func.__name__ == 'f10':
            return -func(individual)
        elif func.__name__ == 'f11':
            return -func(individual)
        elif func.__name__ == 'f12':
            return -func(individual)
        elif func.__name__ == 'f13':
            return -func(individual)
        elif func.__name__ == 'f14':
            return -func(individual)
        elif func.__name__ == 'f15':
            return -func(individual)
        elif func.__name__ == 'f16':
            return -func(individual)
        elif func.__name__ == 'f17':
            return -func(individual)
        elif func.__name__ == 'f18':
            return -func(individual)
        elif func.__name__ == 'f19':
            return -func(individual)
        elif func.__name__ == 'f20':
            return -func(individual)
        elif func.__name__ == 'f21':
            return -func(individual)
        elif func.__name__ == 'f22':
            return -func(individual)
        elif func.__name__ == 'f23':
            return -func(individual)
        elif func.__name__ == 'f24':
            return -func(individual)

    def crossover(self, parent1, parent2):
        child = parent1 + (parent2 - parent1) * 0.5
        return child

    def mutation(self, individual):
        mutation_rate = 0.1
        if random.random() < mutation_rate:
            individual = individual + random.uniform(-1.0, 1.0, self.dim)
        return individual

    def refine_strategy(self, individual):
        if random.random() < 0.4:
            if random.random() < 0.5:
                individual = self.crossover(individual, individual)
            else:
                individual = self.mutation(individual)
        return individual

    def update(self, func):
        self.fitness_history = np.zeros(self.budget)
        for i in range(self.budget):
            individual = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
            individual = self.refine_strategy(individual)
            self.fitness_history[i] = self.evaluate_fitness(func, individual)
            individual = self.refine_strategy(individual)
            self.fitness_history[i] = self.evaluate_fitness(func, individual)
            individual = self.refine_strategy(individual)
            self.fitness_history[i] = self.evaluate_fitness(func, individual)
            individual = self.refine_strategy(individual)
            self.fitness_history[i] = self.evaluate_fitness(func, individual)
            individual = self.refine_strategy(individual)
            self.fitness_history[i] = self.evaluate_fitness(func, individual)
            individual = self.refine_strategy(individual)
            self.fitness_history[i] = self.evaluate_fitness(func, individual)
            individual = self.refine_strategy(individual)
            self.fitness_history[i] = self.evaluate_fitness(func, individual)
        return np.min(self.fitness_history)

# Usage:
budget = 50
dim = 10
func = lambda x: x[0]**2 + x[1]**2
heacombbo = HEACOMBBO(budget, dim)
print(heacombbo.update(func))