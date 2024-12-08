import numpy as np

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness = np.zeros((self.population_size,))

    def __call__(self, func):
        for _ in range(self.budget):
            # Evaluate the fitness of the current population
            self.fitness = func(self.population)

            # Select the fittest individuals
            indices = np.argsort(self.fitness)[:self.population_size//2]
            self.population = self.population[indices]
            self.fitness = self.fitness[indices]

            # Calculate the gradient of the fitness function
            gradient = np.zeros((self.dim,))
            for i in range(self.population_size):
                gradient += (self.fitness[i] - self.fitness[i-1]) * (self.population[i] - self.population[i-1]) / (self.population[i] - self.population[i-1])**2

            # Update the population using evolutionary strategies
            self.population += 0.1 * np.random.normal(0, 1, (self.population_size, self.dim))
            self.population = np.clip(self.population, -5.0, 5.0)

            # Refine the strategy by changing the learning rate and mutation rate with a probability of 0.3
            if np.random.rand() < 0.3:
                self.population *= np.random.uniform(0.9, 1.1, (self.population_size, self.dim))
            if np.random.rand() < 0.3:
                self.population += np.random.normal(0, 0.1, (self.population_size, self.dim))

            # Normalize the population
            self.population = self.population / np.max(np.abs(self.population))
            self.fitness = func(self.population)

# Test the algorithm
def test_evolutionary_gradient():
    def func(x):
        return np.sum(x**2)

    budget = 100
    dim = 10
    algorithm = EvolutionaryGradient(budget, dim)
    algorithm()
    print(algorithm.population)

test_evolutionary_gradient()