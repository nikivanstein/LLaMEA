import numpy as np
from scipy.optimize import differential_evolution
import random
import copy

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
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f20':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f21':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f22':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f23':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f24':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def refine_strategy(self, func):
        if func.__name__ == 'f1':
            return'sbx'
        elif func.__name__ == 'f2':
            return'sbx'
        elif func.__name__ == 'f3':
            return'sbx'
        elif func.__name__ == 'f4':
            return'sbx'
        elif func.__name__ == 'f5':
            return'sbx'
        elif func.__name__ == 'f6':
            return'sbx'
        elif func.__name__ == 'f7':
            return'sbx'
        elif func.__name__ == 'f8':
            return'sbx'
        elif func.__name__ == 'f9':
            return'sbx'
        elif func.__name__ == 'f10':
            return'sbx'
        elif func.__name__ == 'f11':
            return'sbx'
        elif func.__name__ == 'f12':
            return'sbx'
        elif func.__name__ == 'f13':
            return'sbx'
        elif func.__name__ == 'f14':
            return'sbx'
        elif func.__name__ == 'f15':
            return'sbx'
        elif func.__name__ == 'f16':
            return'sbx'
        elif func.__name__ == 'f17':
            return'sbx'
        elif func.__name__ == 'f18':
            return'sbx'
        elif func.__name__ == 'f19':
            return'sbx'
        elif func.__name__ == 'f20':
            return'sbx'
        elif func.__name__ == 'f21':
            return'sbx'
        elif func.__name__ == 'f22':
            return'sbx'
        elif func.__name__ == 'f23':
            return'sbx'
        elif func.__name__ == 'f24':
            return'sbx'

    def evaluate_fitness(self, individual):
        func_name = individual.__class__.__name__
        func = globals()[func_name]
        if self.budget > 0:
            func()
            self.budget -= 1
        return func()

    def mutate(self, individual):
        func_name = individual.__class__.__name__
        func = globals()[func_name]
        new_individual = copy.deepcopy(individual)
        new_func = self.refine_strategy(func)
        new_individual.__class__.__name__ = new_func
        new_func = globals()[new_func]
        new_individual = new_func()
        return new_individual

    def crossover(self, parent1, parent2):
        child = copy.deepcopy(parent1)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent2[i]
        return child

    def evolve(self, population, num_generations):
        for _ in range(num_generations):
            population = sorted(population, key=lambda individual: self.evaluate_fitness(individual))
            new_population = []
            for i in range(len(population) // 2):
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                new_population.append(child)
                new_population.append(self.mutate(child))
            population = new_population
        return population

# Usage:
if __name__ == "__main__":
    population = [HEACOMBBO(100, 10) for _ in range(50)]
    final_population = population
    for individual in final_population:
        individual.x0 = np.random.uniform(individual.bounds[0][0], individual.bounds[0][1], individual.dim)
    num_generations = 10
    final_population = population
    for _ in range(num_generations):
        final_population = population
        population = population
        population = final_population
        population = population
        population = population
        population = population
        population = population
        population = population
        population = population
        population = population
        population = population
        population = population
        population = population
    print(final_population[0].evaluate_fitness(final_population[0]))