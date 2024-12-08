import random
import numpy as np

class AdaptiveBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.best_individual = None
        self.best_fitness = -np.inf

    def __call__(self, func, iterations=100):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + iterations)
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        point = np.random.choice(self.search_space)

        # Evaluate the function at the point
        value = func(point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return point
        else:
            # If the function has been evaluated within the budget, return the point
            return point

        # Update the best individual and fitness if necessary
        if value > self.best_fitness:
            self.best_individual = point
            self.best_fitness = value
            self.best_individual_fitness = func(point)

    def mutate(self, individual):
        # Randomly select a dimension and flip its value
        index = random.randint(0, self.dim - 1)
        individual[index] = 1 - individual[index]

    def __str__(self):
        return f"Adaptive Black Box Optimizer: A novel metaheuristic algorithm that dynamically adjusts its search strategy based on the performance of previous solutions"

# One-line description: "Adaptive Black Box Optimizer: A novel metaheuristic algorithm that dynamically adjusts its search strategy based on the performance of previous solutions"