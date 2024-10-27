import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, mutation_rate, population_size):
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

    def mutate(self, individual, mutation_rate):
        # Randomly select an index in the individual
        idx = random.randint(0, self.dim - 1)

        # Apply the mutation
        individual[idx] = random.uniform(-5.0, 5.0)

        # Check if the mutation is within the search space
        if random.random() < mutation_rate:
            # If not, apply a small perturbation to the individual
            individual[idx] += random.uniform(-0.1, 0.1)

        # Check if the individual has been mutated within the budget
        if individual_evaluations(self.func, individual) < self.budget:
            # If not, return the mutated individual
            return individual
        else:
            # If the individual has been mutated within the budget, return the original individual
            return individual

    def evaluate_fitness(self, individual, func):
        # Evaluate the function at the individual
        value = func(individual)

        # Check if the function has been evaluated within the budget
        if value < 1e-10:  # arbitrary threshold
            # If not, return the current individual as the optimal solution
            return individual
        else:
            # If the function has been evaluated within the budget, return the individual
            return individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"