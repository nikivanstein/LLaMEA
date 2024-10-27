# Description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"
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
        # Select a random individual from the current population
        new_individual = random.choice([i for i in self.population if i!= individual])

        # Generate a new individual by perturbing the current individual
        new_point = individual + np.random.uniform(-1, 1, self.dim)

        # Evaluate the new individual
        new_value = func(new_point)

        # Check if the new individual has been evaluated within the budget
        if new_value < 1e-10:  # arbitrary threshold
            # If not, return the new individual as the mutated solution
            return new_individual
        else:
            # If the new individual has been evaluated within the budget, return the new individual
            return new_point

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

class BlackBoxOptimizerMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0
        self.population = []

    def __call__(self, func):
        # Initialize the population with random individuals
        for _ in range(100):
            self.population.append(func(np.random.rand(self.dim)))

        # Run the algorithm for the specified number of iterations
        for _ in range(100):
            # Select the fittest individual
            fittest_individual = max(self.population, key=func)

            # Generate a new individual by perturbing the fittest individual
            new_individual = fittest_individual + np.random.uniform(-1, 1, self.dim)

            # Evaluate the new individual
            new_value = func(new_individual)

            # Check if the new individual has been evaluated within the budget
            if new_value < 1e-10:  # arbitrary threshold
                # If not, return the new individual as the mutated solution
                return new_individual
            else:
                # If the new individual has been evaluated within the budget, return the new individual
                return new_individual

# Example usage:
if __name__ == "__main__":
    # Create a new Black Box Optimizer with 1000 budget evaluations and 5 dimensions
    optimizer = BlackBoxOptimizerMetaheuristic(1000, 5)

    # Run the algorithm to find the optimal solution
    optimal_individual = optimizer(__call__)

    # Print the result
    print(f"Optimal solution: {optimal_individual}")