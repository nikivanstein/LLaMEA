import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
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

    def mutate(self, individual):
        # Randomly select an individual in the search space
        new_individual = random.choice(self.search_space)

        # Evaluate the new individual
        new_value = self(func, new_individual)

        # Check if the new individual has been evaluated within the budget
        if new_value < 1e-10:  # arbitrary threshold
            # If not, return the new individual
            return new_individual
        else:
            # If the new individual has been evaluated within the budget, return the individual
            return individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

class MutationExp:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        # Evaluate the function for the specified number of times
        num_evaluations = min(self.budget, self.func_evaluations + 1)
        func_evaluations = self.func_evaluations
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

# One-line description: "Meta-Heuristic for Optimization: Combining Random Search and Function Evaluation"

def main():
    # Create a budget and dimension
    budget = 1000
    dim = 10

    # Create a Black Box Optimizer
    optimizer = BlackBoxOptimizer(budget, dim)

    # Create a MutationExp object
    mutation_exp = MutationExp(budget, dim)

    # Initialize the selected solution
    selected_solution = "BlackBoxOptimizer"

    # Call the optimize function
    for _ in range(100):
        # Generate a new individual
        new_individual = mutation_exp(optimized_individual)

        # Evaluate the new individual
        new_value = optimizer(new_individual)

        # Check if the new individual has been evaluated within the budget
        if new_value < 1e-10:  # arbitrary threshold
            # If not, update the selected solution
            selected_solution = new_individual
            break

    # Print the selected solution
    print(f"Selected solution: {selected_solution}")

if __name__ == "__main__":
    main()