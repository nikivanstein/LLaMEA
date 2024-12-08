import numpy as np
from scipy.stats import norm

class EvolutionStrategy:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.mutation_rate = 0.1
        self.mutation_std = 1.0
        self.mean = np.zeros(self.dim)
        self.std = np.ones(self.dim)
        self.population = np.random.normal(self.mean, self.std, (self.population_size, self.dim))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the function at the current population
            evaluations = np.array([func(x) for x in self.population.flatten()]).reshape(self.population_size, self.dim)

            # Select the best individual
            best_idx = np.argmax(evaluations, axis=0)
            best_individual = self.population[best_idx]

            # Compute the probability of mutation for each dimension
            mutation_probabilities = norm.pdf(np.abs(evaluations), scale=self.mutation_std)

            # Refine the strategy with probability 0.3
            if np.random.rand() < 0.3:
                for i in range(self.dim):
                    if np.random.rand() < mutation_probabilities[i]:
                        # Perform mutation
                        self.population[:, i] += np.random.normal(0, self.mutation_std, self.population_size)
                        self.std[i] *= 1.1  # Increase the standard deviation for the next iteration

            # Update the mean
            self.mean = np.mean(self.population, axis=0)

            # Update the population
            self.population = np.random.normal(self.mean, self.std, (self.population_size, self.dim))

            # Print the best individual and its evaluation
            print(f"Best individual: {best_individual}, Evaluation: {evaluations[best_idx]}")

# Test the algorithm
def func(x):
    return np.sum(x**2)

es = EvolutionStrategy(budget=100, dim=10)
es("func")