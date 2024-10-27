import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveMutation:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, size=(budget, dim))
        self.fitness = np.zeros(budget)

    def __call__(self, func):
        for i in range(self.budget):
            # Evaluate the function at the current population
            self.fitness[i] = func(self.population[i])

            # Calculate the probability of mutation for each dimension
            mutation_prob = np.random.uniform(0.0, 1.0, size=self.dim)
            mutation_prob[mutation_prob < 0.3] = 0.0

            # Perform mutation on the fittest individual
            fittest_idx = np.argmin(self.fitness)
            mutation_idx = np.random.choice(self.dim, p=mutation_prob)
            self.population[fittest_idx, mutation_idx] += np.random.uniform(-0.5, 0.5)

            # Ensure bounds are not exceeded
            self.population[fittest_idx, :] = np.clip(self.population[fittest_idx, :], -5.0, 5.0)

        # Perform differential evolution optimization
        res = differential_evolution(func, [(-5.0, 5.0)] * self.dim, x0=self.population, maxiter=self.budget)
        self.population = res.x

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

budget = 50
dim = 2
optimizer = AdaptiveMutation(budget, dim)
optimizer(func)