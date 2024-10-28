import numpy as np
import random

class EvolutionaryTreeBasedOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.num_iterations = self.budget
        self.tree_size = 20
        self.branching_probability = 0.5
        self.mutation_rate = 0.1
        self.pbest = np.zeros((self.population_size, self.dim))
        self.gbest = np.zeros(self.dim)
        self.tree = np.zeros((self.population_size, self.tree_size, self.dim))

    def __call__(self, func):
        for _ in range(self.num_iterations):
            # Initialize tree
            for i in range(self.population_size):
                self.tree[i, 0, :] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                self.pbest[i, :] = self.tree[i, 0, :]
                self.gbest[:] = self.tree[i, 0, :]

            # Main loop
            for _ in range(self.num_iterations):
                # Evaluate tree
                values = func(self.tree)

                # Update pbest and gbest
                for i in range(self.population_size):
                    if values[i, 0] < self.pbest[i, 0]:
                        self.pbest[i, :] = self.tree[i, 0, :]
                    if values[i, 0] < self.gbest[0]:
                        self.gbest[:] = self.tree[i, 0, :]

                # Branching and mutation
                for i in range(self.population_size):
                    if random.random() < self.branching_probability:
                        # Select parent
                        parent = random.randint(0, self.population_size - 1)

                        # Select child
                        child = np.random.choice([0, 1])

                        # Create new branch
                        self.tree[i, child, :] = (self.tree[parent, 0, :] + self.tree[i, 0, :]) / 2
                        if random.random() < self.mutation_rate:
                            self.tree[i, child, :] += np.random.uniform(-1.0, 1.0, self.dim)

                    # Mutation
                    if random.random() < self.mutation_rate:
                        self.tree[i, 0, :] += np.random.uniform(-1.0, 1.0, self.dim)

            # Return the best solution
            return self.gbest[0]

# Example usage:
def func(x):
    return x[0]**2 + x[1]**2

optimizer = EvolutionaryTreeBasedOptimization(budget=100, dim=2)
result = optimizer(func)
print(result)