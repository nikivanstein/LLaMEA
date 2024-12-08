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
            # Generate a new individual by refining the current one
            new_individual = self.evaluate_fitness(self.func(self.func(self.search_space[np.random.randint(0, self.search_space.shape[0]), :])), self.budget / 2)
            
            # Refine the individual with a probability of 0.05
            if random.random() < 0.05:
                new_individual = self.evaluate_fitness(self.func(self.func(new_individual, self.budget / 2)), self.budget / 2)
            
            # Update the search space with the new individual
            self.search_space = np.vstack((self.search_space, new_individual))
            self.search_space = np.delete(self.search_space, 0, axis=0)
            
            # Evaluate the new individual using the original function
            new_individual = self.func(new_individual)
            
            # Check if the new individual is better than the current best individual
            best_individual = self.func(self.search_space[np.random.randint(0, self.search_space.shape[0]), :])
            if new_individual < best_individual:
                # Update the best individual
                best_individual = new_individual
                
            # Return the best individual found
            return best_individual

    def evaluate_fitness(self, individual, budget):
        # Evaluate the fitness of the individual using the original function
        return self.func(individual)