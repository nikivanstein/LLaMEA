import random
import numpy as np

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
        new_individual = self.evaluate_fitness(best_func)
        self.search_space = [x for x in self.search_space if x not in new_individual]

        return new_individual

    def evaluate_fitness(self, individual):
        # Select a fitness function that evaluates the individual's fitness
        fitness_func = lambda x: np.mean([func(x) for func in [self.budget] + [self.dim] + [x for x in self.search_space if x not in individual]])

        # Evaluate the individual's fitness
        fitness = fitness_func(individual)

        # Refine the strategy by changing 45% of the individual's lines
        for i in range(len(individual)):
            if random.random() < 0.45:
                individual[i] = random.choice([x for x in individual[i] if x!= individual[i]])

        return fitness

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 