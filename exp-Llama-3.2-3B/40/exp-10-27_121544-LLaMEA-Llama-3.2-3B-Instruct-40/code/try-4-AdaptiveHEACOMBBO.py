import numpy as np
import random
import code
import operator

class AdaptiveHEACOMBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.bounds = [(-5.0, 5.0)] * dim
        self.x0 = np.random.uniform(self.bounds[0][0], self.bounds[0][1], self.dim)
        self.population = [self.x0]
        self.fitnesses = [self.f(self.x0)]
        self.strategies = [0.5]

    def __call__(self, func):
        if self.budget == 0:
            return np.nan, np.nan

        if func.__name__ == 'f1':
            return self.adaptive_differential_evolution(func, self.bounds)
        elif func.__name__ == 'f2':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0)
        elif func.__name__ == 'f3':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5)
        elif func.__name__ == 'f4':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0)
        elif func.__name__ == 'f5':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx')
        elif func.__name__ == 'f6':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5)
        elif func.__name__ == 'f7':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5)
        elif func.__name__ == 'f8':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5)
        elif func.__name__ == 'f9':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1')
        elif func.__name__ == 'f10':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform')
        elif func.__name__ == 'f11':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform')
        elif func.__name__ == 'f12':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f13':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform')
        elif func.__name__ == 'f14':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform')
        elif func.__name__ == 'f15':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f16':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f17':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f18':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f20':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f21':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f22':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f23':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f24':
            return self.adaptive_differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def adaptive_differential_evolution(self, func, bounds, x0=None, tol=1e-5, x0_init=None, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform'):
        if x0 is None:
            x0 = np.random.uniform(bounds[0][0], bounds[0][1], self.dim)
        if x0_init is not None:
            x0 = x0_init

        population = [x0]
        fitnesses = [self.f(func, x0)]

        for _ in range(self.budget - 1):
            new_individuals = []
            new_fitnesses = []
            for individual in population:
                for _ in range(int(cxpb * len(population))):
                    new_individual = individual + np.random.uniform(-1, 1, self.dim)
                    new_individual = [min(max(i, bounds[i][0]), bounds[i][1]) for i in range(self.dim)]
                    new_individuals.append(new_individual)
                    new_fitness = self.f(func, new_individual)
                    new_fitnesses.append(new_fitness)

            population = population + new_individuals
            fitnesses = fitnesses + new_fitnesses

            # Adaptive strategy selection
            strategies = [0.4, 0.3, 0.3]
            strategy_idx = np.random.choice(len(strategies), p=strategies)
            strategy = strategies[strategy_idx]

            # Adaptive mutation
            for i in range(len(population)):
                if np.random.rand() < strategy:
                    mutation = np.random.uniform(-1, 1, self.dim)
                    population[i] = [min(max(i, bounds[i][0]), bounds[i][1]) for i in range(self.dim)]

        # Select the best individual
        best_individual = min(population, key=self.f)
        best_fitness = self.f(func, best_individual)

        return best_individual, best_fitness

    def f(self, func, x):
        return func(x)

# Usage
if __name__ == "__main__":
    bbb = AdaptiveHEACOMBBO(budget=100, dim=10)
    for func_name in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24']:
        func = globals()[func_name]
        result = bbb(func)
        print(f"Function: {func_name}, Result: {result}")