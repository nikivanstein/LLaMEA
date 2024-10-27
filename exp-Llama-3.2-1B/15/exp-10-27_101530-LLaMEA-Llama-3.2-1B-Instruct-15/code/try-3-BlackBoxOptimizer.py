import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Evaluate the function at the point
            func_value = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return point
        # If the budget is reached, return the best point found so far
        return self.search_space[0], self.search_space[1]

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

    def __str__(self, budget):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization (Budget: {budget})"

    def __str__dim(self, dim):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization (Dimension: {dim})"

    def mutate(self, new_individual):
        # Refine the strategy by changing a random line of the selected solution
        line_index = random.randint(0, self.dim - 1)
        new_line = random.uniform(self.search_space[0], self.search_space[1])
        return new_individual[:line_index] + [new_line] + new_individual[line_index + 1:]

    def __str__(self, budget, dim):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization (Budget: {budget}, Dimension: {dim})"

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
def black_box_optimization(budget, dim):
    optimizer = BlackBoxOptimizer(budget, dim)
    for _ in range(100):
        new_individual = optimizer()
        print(optimizer)
        if optimizer == BlackBoxOptimizer("Novel Metaheuristic Algorithm for Black Box Optimization"):
            print("Solution found!")
            return new_individual
    print("No solution found after 100 iterations.")
    return None

# Test the algorithm
budget = 100
dim = 10
new_individual = black_box_optimization(budget, dim)
if new_individual is not None:
    print(new_individual)