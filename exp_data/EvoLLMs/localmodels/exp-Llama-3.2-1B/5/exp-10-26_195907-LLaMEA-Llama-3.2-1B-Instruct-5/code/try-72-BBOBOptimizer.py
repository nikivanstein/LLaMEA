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
            for _ in range(self.budget):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            if random.random() < 0.05:  # Refine the strategy
                new_individual = self.evaluate_fitness(new_individual)
                updated_individual = self.f(individual, new_individual, self.budget)
                self.search_space = updated_individual

    def evaluate_fitness(self, individual):
        return self.func(individual)

    def f(self, individual, new_individual, budget):
        # Calculate the fitness of the new individual
        fitness = self.func(new_individual)
        # Calculate the fitness of the current individual
        current_fitness = self.func(individual)
        # Return the fitness of the new individual
        return fitness - current_fitness * 0.05  # Apply the refinement strategy