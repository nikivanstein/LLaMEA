import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def __call__(self, func):
        def evaluate_fitness(individual):
            # Refine the strategy by changing the individual lines
            new_individual = individual + np.random.uniform(-0.1, 0.1, self.dim)
            # Evaluate the function at the new individual
            value = func(new_individual)
            # Check if the new individual is within the bounds
            if -5.0 <= new_individual[0] <= 5.0 and -5.0 <= new_individual[1] <= 5.0:
                # If the new individual is within bounds, update the function value
                self.func_evals += 1
                return value
            else:
                # If the new individual is not within bounds, return the original value
                return func(individual)
        return evaluate_fitness

# Initialize the selected solution
selected_solution = BlackBoxOptimizer(budget=1000, dim=10)

# Define the function to optimize
def func(x):
    return np.sin(x)

# Optimize the function using the selected solution
optimized_func = selected_solution(func)

# Print the score
print(f"Score: {optimized_func(np.random.uniform(-5.0, 5.0, 10))}")