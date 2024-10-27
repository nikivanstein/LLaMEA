import random
import numpy as np

class MetaBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.current_individual = None
        self.current_point = None
        self.current_fitness = None
        self.mutation_rate = 0.01
        self.crossover_rate = 0.5

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        self.func_evaluations += num_evaluations

        # Generate a random point in the search space
        self.current_individual = np.random.choice(self.search_space)
        self.current_point = self.current_individual
        self.current_fitness = func(self.current_point)

        # Evaluate the function at the point
        value = func(self.current_point)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current point as the optimal solution
            return self.current_point
        else:
            # If the function has been evaluated within the budget, return the point
            return self.current_point

    def mutate(self):
        if random.random() < self.mutation_rate:
            # Randomly select two points in the search space
            point1 = np.random.choice(self.search_space)
            point2 = np.random.choice(self.search_space)

            # Generate a new point by crossover
            self.current_point = (self.current_point + point1 * self.crossover_rate + point2 * (1 - self.crossover_rate)) / 2

    def crossover(self):
        if random.random() < self.crossover_rate:
            # Randomly select a point in the search space
            point1 = np.random.choice(self.search_space)

            # Generate a new point by mutation
            self.current_point = point1
            self.mutate()

# One-line description: "Meta-Black Box Optimizer: A hybrid algorithm combining random search and function evaluation to efficiently solve black box optimization problems"

# Example usage:
optimizer = MetaBlackBoxOptimizer(100, 5)
problem = RealSingleObjectiveProblem(1, "Sphere", 5.0)
solution = optimizer(problem)
print(solution)