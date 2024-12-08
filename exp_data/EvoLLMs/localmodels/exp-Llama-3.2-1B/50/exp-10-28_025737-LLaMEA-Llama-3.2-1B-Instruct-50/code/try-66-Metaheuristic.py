# Description: Novel Metaheuristic Algorithm for Black Box Optimization (NMA)
# Code: 
# ```python
import numpy as np
import random

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

class NMA(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
        self.population_size = 100

    def __call__(self, func):
        population = [self.__call__(func) for _ in range(self.population_size)]
        return self.select(population, 0.45)

    def select(self, population, p):
        # Select parents using tournament selection
        parents = random.sample(population, int(len(population) * p))

        # Select the best individual in each tournament
        winners = []
        for parent in parents:
            winner = max(set(parent), key=parent.count)
            winners.append(winner)

        # Select the best individual from the winners
        best_individual = max(set(winners), key=winners.count)
        return best_individual

    def mutate(self, individual):
        # Mutate the individual by changing one gene at random
        mutated_individual = individual.copy()
        mutated_individual[random.randint(0, self.dim-1)] = random.uniform(-5.0, 5.0)
        return mutated_individual

# Evaluate the function on the BBOB test suite
# ```python
# func = lambda x: x**2
# func_values = [func(x) for x in np.random.uniform(-10, 10, size=100)]
# print(func_values)

# Initialize the algorithm
nma = NMA(100, 10)
# ```python