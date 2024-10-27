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
            for _ in range(min(self.budget, self.dim)):
                x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
                if np.linalg.norm(func(x)) < self.budget / 2:
                    return x
            x = self.search_space[np.random.randint(0, self.search_space.shape[0])]
            self.search_space = np.vstack((self.search_space, x))
            self.search_space = np.delete(self.search_space, 0, axis=0)

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

    def mutate(self, individual):
        if random.random() < 0.05:
            dim = min(self.dim, 2 * random.randint(1, self.dim))
            individual = np.vstack((individual, np.random.uniform(-5.0, 5.0, size=(dim,))))
        return individual

# Initialize the selected solution
BBOBOptimizer(100, 10)

# Create a new individual
new_individual = BBOBOptimizer(100, 10).evaluate_fitness(np.array([[-5.0, -5.0], [-5.0, -5.0], [-5.0, -5.0]]))

# Call the function
print(BBOBOptimizer(100, 10)('Novel Metaheuristic Algorithm for Black Box Optimization'))