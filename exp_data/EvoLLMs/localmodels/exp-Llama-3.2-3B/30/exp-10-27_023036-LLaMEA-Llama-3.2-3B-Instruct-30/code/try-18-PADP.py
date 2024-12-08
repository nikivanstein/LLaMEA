import numpy as np
from scipy.optimize import minimize

class PADP:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.population = []
        self.fitness_values = []

    def __call__(self, func):
        # Initialize the population
        for _ in range(self.population_size):
            x0 = np.random.uniform(-5.0, 5.0, self.dim)
            self.population.append(x0)
            self.fitness_values.append(func(x0))

        # Perform adaptive dynamic programming
        for i in range(self.budget):
            # Select the best individual
            idx = np.argmin(self.fitness_values)
            best_individual = self.population[idx]

            # Refine the best individual's strategy
            for j in range(self.population_size):
                # With probability 0.3, change one line of the best individual
                if np.random.rand() < 0.3:
                    idx = np.random.randint(0, self.dim)
                    best_individual[idx] += np.random.uniform(-0.5, 0.5)
                # Evaluate the new individual
                new_individual = best_individual.copy()
                new_individual[idx] = np.clip(new_individual[idx], -5.0, 5.0)
                new_fitness = func(new_individual)
                self.population[j] = new_individual
                self.fitness_values[j] = new_fitness

                # Replace the worst individual with the new one
                idx = np.argmin(self.fitness_values)
                self.population[idx] = new_individual
                self.fitness_values[idx] = new_fitness

        # Return the best individual
        best_individual = np.min(self.population, axis=0)
        return best_individual, np.min(self.fitness_values)

# Example usage
def func(x):
    return np.sum(x**2)

budget = 100
dim = 5
padp = PADP(budget, dim)
best_individual, best_fitness = padp(func)
print("Best individual:", best_individual)
print("Best fitness:", best_fitness)