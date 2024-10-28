import random
import numpy as np
import copy

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Evaluate the function a limited number of times
        num_evals = min(self.budget, len(func(self.search_space)))
        func_values = [func(x) for x in random.sample(self.search_space, num_evals)]

        # Select the best function value
        best_func = max(set(func_values), key=func_values.count)

        # Update the search space
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

    def mutate(self, individual):
        # Refine the strategy by changing 10% of the individual's lines
        new_individual = copy.deepcopy(individual)
        for i in range(len(new_individual)):
            if random.random() < 0.1:
                new_individual[i] = random.uniform(self.search_space[i][0], self.search_space[i][1])
        return new_individual

    def crossover(self, parent1, parent2):
        # Perform a single-point crossover to create a new individual
        new_individual = copy.deepcopy(parent1)
        crossover_point = random.randint(1, len(new_individual) - 1)
        new_individual[crossover_point] = parent2[crossover_point]
        return new_individual

# One-line description with the main idea
# Novel Metaheuristic Algorithm for Black Box Optimization: 
# This algorithm uses a novel combination of mutation and crossover to refine the strategy, 
# by changing 10% of the individual's lines and performing a single-point crossover 
# to create a new individual, while also keeping track of the best function value and 
# updating the search space as needed.