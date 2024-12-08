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
            self.search_space = self.search_space[:self.budget]

def novel_metaheuristic(dim, budget):
    # Novel Metaheuristic Algorithm for Black Box Optimization
    # Description: A novel metaheuristic algorithm for black box optimization
    # Code: 
    ```python
    # Initialize the population of algorithms
    algorithms = [
        BBOBOptimizer(100, dim),  # 100 individuals
        BBOBOptimizer(50, dim),   # 50 individuals
        BBOBOptimizer(20, dim)    # 20 individuals
    ]

    # Evaluate the fitness of each algorithm
    for algorithm in algorithms:
        algorithm.func(np.random.rand(dim))  # Evaluate the fitness of the algorithm

    # Select the algorithm with the best fitness
    selected_algorithm = min(algorithms, key=lambda x: x.func(np.random.rand(dim)))

    # Refine the strategy of the selected algorithm
    selected_algorithm.budget = min(selected_algorithm.budget, 100)  # Limit the budget
    selected_algorithm.dim = min(selected_algorithm.dim, 10)  # Limit the dimensionality

    return selected_algorithm

# Initialize the algorithm
selected_algorithm = novel_metaheuristic(3, 50)

# Evaluate the fitness of the selected algorithm
selected_algorithm.func(np.random.rand(3, 10))  # Evaluate the fitness of the selected algorithm