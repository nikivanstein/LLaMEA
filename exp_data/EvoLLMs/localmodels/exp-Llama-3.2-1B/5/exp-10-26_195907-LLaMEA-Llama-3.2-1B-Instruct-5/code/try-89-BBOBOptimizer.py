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

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

    def update_individual(self, individual, budget):
        if budget <= 0:
            raise ValueError("Invalid budget")
        if len(individual)!= self.dim:
            raise ValueError("Invalid individual size")
        if np.linalg.norm(self.func(individual)) >= self.budget / 2:
            return individual
        # Refine strategy by changing individual lines of the selected solution
        if random.random() < 0.05:
            # Increase the step size of the current line
            self.func(individual) += random.uniform(-0.1, 0.1)
            # Change the line to the one with the smaller fitness value
            min_index = np.argmin(self.func(individual))
            individual[min_index] += random.uniform(-0.1, 0.1)
        return individual