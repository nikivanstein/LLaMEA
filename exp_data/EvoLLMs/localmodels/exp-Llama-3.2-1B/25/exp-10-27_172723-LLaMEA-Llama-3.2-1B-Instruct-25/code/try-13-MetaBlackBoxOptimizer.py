import random
import numpy as np
import copy

class MetaBlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, mutation_rate, epsilon):
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
        # Randomly select a mutation point in the individual
        mutation_point = random.randint(0, self.dim - 1)

        # Swap the element at the mutation point with a random element from the search space
        mutated_individual = copy.deepcopy(individual)
        mutated_individual[mutation_point], mutated_individual[mutation_point + random.randint(0, self.dim - 1)] = mutated_individual[mutation_point + random.randint(0, self.dim - 1)], mutated_individual[mutation_point]

        # Evaluate the mutated individual
        mutated_value = self.__call__(mutated_individual, 10, 0.1)

        # Check if the mutated individual is better than the original individual
        if mutated_value > self.__call__(individual, 10, 0.1):
            # If yes, return the mutated individual
            return mutated_individual
        else:
            # If not, return the original individual
            return individual

# One-line description: "Meta-Black Box Optimizer: A novel algorithm that combines metaheuristic search with function evaluation to efficiently solve black box optimization problems"