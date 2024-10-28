import numpy as np
from scipy.optimize import differential_evolution
import random

class HybridEvolutionaryAlgorithm:
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
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f18':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f19':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f20':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f21':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f22':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f23':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform')
        elif func.__name__ == 'f24':
            return differential_evolution(func, self.bounds, x0=self.x0, tol=1e-5, x0_init=self.x0, strategy='sbx', cxpb=0.5, mu=0.5, w=0.5, strategy='rand1', scaling='uniform', scaling='log-uniform', scaling='uniform', scaling='uniform', scaling='uniform')

    def mutate(self, individual):
        if random.random() < 0.4:
            for i in range(self.dim):
                if random.random() < 0.5:
                    individual[i] = self.bounds[i][0] + random.uniform(0, self.bounds[i][1] - self.bounds[i][0])
                else:
                    individual[i] = self.bounds[i][1] - random.uniform(0, self.bounds[i][1] - self.bounds[i][0])
        return individual

    def crossover(self, parent1, parent2):
        child = np.zeros(self.dim)
        for i in range(self.dim):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def hybrid_evolutionary_algorithm(self, func):
        population = [self.x0]
        for _ in range(self.budget):
            population.append(self.mutate(self.crossover(population[-1], population[np.random.choice(len(population))])))
        best_individual = min(population, key=lambda individual: func(individual))
        return best_individual, func(best_individual)

# Example usage
if __name__ == "__main__":
    func = lambda x: x[0]**2 + x[1]**2
    algorithm = HybridEvolutionaryAlgorithm(budget=100, dim=2)
    best_individual, best_function_value = algorithm.hybrid_evolutionary_algorithm(func)
    print(f"Best individual: {best_individual}")
    print(f"Best function value: {best_function_value}")