# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        while True:
            # Select the best individual from the current population
            best_individual = np.argmax(self.evaluate_fitness(self.population))

            # Refine the strategy by changing the lines that refine the individual's strategy
            # to minimize the fitness
            for i in range(self.dim):
                # Calculate the new fitness for the individual
                new_fitness = self.evaluate_fitness(best_individual + i * 0.1 * np.random.uniform(-5.0, 5.0))

                # If the new fitness is better, update the individual
                if new_fitness < self.evaluate_fitness(best_individual):
                    best_individual = best_individual + i * 0.1 * np.random.uniform(-5.0, 5.0)

            # Add a new individual to the population
            new_individual = np.random.uniform(self.search_space.shape[0], self.search_space.shape[0] * 2)
            self.population = np.vstack((self.population, [best_individual, new_individual]))

            # If the population has reached the budget, reset it
            if len(self.population) == self.budget:
                self.population = np.delete(self.population, 0, axis=0)

    def evaluate_fitness(self, individual):
        return self.func(individual)

# Initialize the optimizer with a budget of 100 and a dimension of 10
optimizer = BBOBOptimizer(100, 10)

# Call the optimizer with the black box function
func = lambda x: np.sum(x)
optimizer(func)