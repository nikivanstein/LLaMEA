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
        return f"BlackBoxOptimizer: Novel Metaheuristic Algorithm for Black Box Optimization"

    def mutate(self, new_individual):
        # Refine the strategy by changing the individual lines
        if random.random() < 0.15:
            # Change the function value
            new_func_value = func(new_individual)
            # Change the point
            new_point = (random.uniform(self.search_space[0], self.search_space[1]), random.uniform(self.search_space[0], self.search_space[1]))
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the point is within the budget
            if self.func_evaluations < self.budget:
                # If not, return the point
                return new_point
            # If the point is within the budget, return the new individual
            return new_individual
        # If the strategy is not refined, return the original individual
        return new_individual

# Define a simple function to evaluate
def func(individual):
    return individual[0]**2 + individual[1]**2

# Define the mutation function
def mutate_func(individual):
    return individual[0] + random.uniform(-1, 1), individual[1] + random.uniform(-1, 1)

# Create an instance of the BlackBoxOptimizer
optimizer = BlackBoxOptimizer(1000, 10)

# Initialize the population
population = [optimizer.__call__(func) for _ in range(100)]

# Evolve the population
while population:
    # Select the fittest individual
    fittest_individual = population[0]
    # Generate a new individual
    new_individual = fittest_individual.__str__()
    # Mutate the new individual
    new_individual = optimizer.mutate(new_individual)
    # Add the new individual to the population
    population.append(new_individual)

# Print the final best individual
best_individual = max(population, key=func)
print(best_individual)