import numpy as np
from scipy.optimize import differential_evolution

class EvoDiff:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        return np.random.uniform(-5.0, 5.0, self.dim) + np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        # Evaluate the black box function with the current population
        func_values = np.array([func(x) for x in self.population])

        # Select the fittest solutions
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]

        # Evolve the population using evolutionary differential evolution
        result = differential_evolution(self.budget, [(x, func(x)) for x in self.population], 
                                       initial=self.population, bounds=[(-5.0, 5.0), (-np.inf, np.inf)], 
                                       n_iter=100, x0=self.population[:])

        # Refine the strategy by changing the mutation rate
        if result.success:
            mutation_rate = 0.1
            self.population = np.random.uniform(-5.0, 5.0, self.dim) + np.random.normal(0, 1, self.dim)
            for _ in range(self.budget):
                mutated_parents = self.population.copy()
                for _ in range(self.population_size):
                    if np.random.rand() < mutation_rate:
                        mutated_parents[_] += np.random.normal(0, 1, self.dim)
                self.population = np.concatenate((self.population, mutated_parents), axis=0)
                self.population = np.concatenate((self.population, self.population[:self.population_size]), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

# One-line description with main idea
# EvoDiff: A novel evolutionary differential evolution algorithm that leverages the concept of evolutionary differences to optimize black box functions.