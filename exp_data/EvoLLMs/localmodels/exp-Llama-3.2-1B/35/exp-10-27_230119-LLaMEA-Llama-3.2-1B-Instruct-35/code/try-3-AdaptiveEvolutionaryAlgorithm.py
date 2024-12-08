import numpy as np
from scipy.optimize import minimize

class AdaptiveEvolutionaryAlgorithm:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01, sampling_rate=0.5):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.sampling_rate = sampling_rate
        self.population = np.random.uniform(-5.0, 5.0, size=(population_size, dim))
        self.best_individual = None
        self.best_score = float('inf')

    def __call__(self, func):
        for _ in range(self.budget):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(self.population)
            if np.any(best_func!= func(self.population)):
                self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(self.population)):
                break
        return func(self.population)

    def adaptive_sampling(self, func):
        num_evaluations = 0
        while num_evaluations < self.budget:
            func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            best_func = func(func_evals)
            if np.any(best_func!= func(func_evals)):
                func_evals = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            if np.all(best_func == func(func_evals)):
                break
            num_evaluations += 1
        return func_evals

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            dim = self.dim
            if dim > 1:
                individual[dim-1] = np.random.uniform(-5.0, 5.0)
            return individual
        return individual

    def evaluate(self, individual):
        return minimize(lambda x: func(x), individual)[1]

    def update(self, individual, func):
        best_func = func(self.population)
        if np.any(best_func!= func(self.population)):
            self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        if np.all(best_func == func(self.population)):
            self.best_individual = individual
            self.best_score = self.evaluate(individual)
        return func(self.population)

# Test the algorithm
def func(x):
    return np.sum(x**2)

algorithm = AdaptiveEvolutionaryAlgorithm(budget=1000, dim=2)
best_individual = algorithm adaptive_sampling(func)
best_individual = algorithm.update(best_individual, func)
print("Best Individual:", best_individual)
print("Best Score:", algorithm.best_score)