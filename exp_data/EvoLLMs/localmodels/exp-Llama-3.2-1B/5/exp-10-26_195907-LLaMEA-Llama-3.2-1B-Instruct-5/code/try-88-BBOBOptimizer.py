import random
import numpy as np

class BBOBOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, size=(dim, 2))
        self.func = lambda x: np.sum(x)

    def __call__(self, func):
        def evaluate_fitness(individual):
            return self.func(individual)

        def __evaluate_fitness(individual, budget):
            if budget <= 0:
                raise ValueError("Budget must be greater than 0")

            while True:
                for _ in range(budget):
                    x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                    if np.linalg.norm(func(x)) < budget / 2:
                        return individual
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                self.search_space = np.vstack((self.search_space, x))
                self.search_space = np.delete(self.search_space, 0, axis=0)

        fitness = evaluate_fitness(self.search_space[np.random.randint(0, self.search_space.shape[0])])
        return fitness

def novel_metaheuristic_algorithm(budget, dim):
    optimizer = BBOBOptimizer(budget, dim)
    return optimizer.__call__(optimizer.func)

# Test the algorithm
print(novel_metaheuristic_algorithm(1000, 10))