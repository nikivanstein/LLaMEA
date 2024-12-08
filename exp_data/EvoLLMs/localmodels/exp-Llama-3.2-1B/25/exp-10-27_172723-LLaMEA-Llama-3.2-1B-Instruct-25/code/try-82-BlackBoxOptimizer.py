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
        # Select a random mutation point within the search space
        mutation_point = np.random.choice(self.search_space)

        # Create a new individual by swapping the mutation point with a random point in the search space
        new_individual = individual.copy()
        new_individual[mutation_point], new_individual[np.random.randint(len(new_individual))] = new_individual[np.random.randint(len(new_individual))], new_individual[mutation_point]

        # Evaluate the new individual
        new_value = func(new_individual)

        # Check if the new individual has been evaluated within the budget
        if new_value < 1e-10:  # arbitrary threshold
            # If not, return the new individual as the mutated solution
            return new_individual
        else:
            # If the new individual has been evaluated within the budget, return the individual
            return individual

# One-line description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"

class BlackBoxOptimizerMetaheuristic:
    def __init__(self, budget, dim):
        self.optimizer = BlackBoxOptimizer(budget, dim)

    def __call__(self, func):
        # Initialize the population with random individuals
        population = [self.optimizer.__call__(func) for _ in range(100)]

        # Evolve the population for a specified number of generations
        for _ in range(100):
            # Select the fittest individuals to reproduce
            parents = random.sample(population, len(population) // 2)

            # Create a new population by breeding the parents
            new_population = [self.optimizer.__call__(func) for func in parents]

            # Mutate the new population
            new_population = [self.optimizer.mutate(individual) for individual in new_population]

            # Replace the old population with the new population
            population = new_population

        # Evaluate the final population
        final_population = [self.optimizer.__call__(func) for func in population]

        # Return the fittest individual
        return min(final_population, key=lambda individual: self.optimizer.func_evaluations)

# Description: "Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation"
# Code: 
# ```python
# Black Box Optimizer: A novel metaheuristic algorithm that efficiently solves black box optimization problems using a combination of random search and function evaluation
# 
# class BlackBoxOptimizerMetaheuristic:
#     def __init__(self, budget, dim):
#         self.optimizer = BlackBoxOptimizer(budget, dim)

#     def __call__(self, func):
#         # Initialize the population with random individuals
#         population = [self.optimizer.__call__(func) for _ in range(100)]

#         # Evolve the population for a specified number of generations
#         for _ in range(100):
#             # Select the fittest individuals to reproduce
#             parents = random.sample(population, len(population) // 2)

#             # Create a new population by breeding the parents
#             new_population = [self.optimizer.__call__(func) for func in parents]

#             # Mutate the new population
#             new_population = [self.optimizer.mutate(individual) for individual in new_population]

#             # Replace the old population with the new population
#             population = new_population

#         # Evaluate the final population
#         final_population = [self.optimizer.__call__(func) for func in population]

#         # Return the fittest individual
#         return min(final_population, key=lambda individual: self.optimizer.func_evaluations)

# ```