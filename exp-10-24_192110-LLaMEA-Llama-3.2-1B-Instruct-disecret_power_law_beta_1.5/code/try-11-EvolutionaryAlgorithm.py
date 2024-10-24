import numpy as np
from scipy.optimize import differential_evolution

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitness_scores = np.zeros(self.population_size)

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds():
            return (self.population.min(), self.population.max())

        result = differential_evolution(objective, bounds, args=(func,), x0=self.population, maxiter=self.budget)
        self.population = result.x
        self.fitness_scores = np.array([func(x) for x in self.population])
        return self.fitness_scores

    def get_best_solution(self):
        return self.population[np.argmax(self.fitness_scores)]

# Test the algorithm
func = lambda x: x**2
algorithm = EvolutionaryAlgorithm(10, 10)
print(algorithm.__call__(func))