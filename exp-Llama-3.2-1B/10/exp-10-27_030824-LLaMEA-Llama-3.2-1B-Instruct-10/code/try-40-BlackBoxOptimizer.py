import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def mutate(self, individual):
        # Select a random individual from the population
        new_individual = random.choice([i for i in self.population if i is not None])
        
        # If the new individual is not None, mutate it
        if new_individual is not None:
            # Generate a random number between 0 and 1
            r = np.random.rand()
            # If the random number is less than 0.1, mutate the individual
            if r < 0.1:
                # Select a random dimension from the search space
                dim = random.randint(0, self.dim - 1)
                # Mutate the individual in the selected dimension
                new_individual[dim] = np.random.uniform(self.search_space[dim], self.search_space[dim] + 0.1)
            # Return the mutated individual
            return new_individual
        else:
            # If the new individual is None, return None
            return None

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

class NovelMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)

    def __call__(self, func):
        # Optimize the function using the BlackBoxOptimizer
        return self.optimizer(func)

# Example usage:
if __name__ == "__main__":
    # Define a black box function
    def func(x):
        return np.sin(x)

    # Create an instance of the optimizer
    optimizer = NovelMetaheuristicOptimizer(100, 10)

    # Optimize the function 100 times
    for _ in range(100):
        # Optimize the function using the optimizer
        optimized_function = optimizer(func)
        print(f"Optimized function: {optimized_function}")