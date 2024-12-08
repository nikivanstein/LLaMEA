# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
import math

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

        # Refine the strategy
        if len(self.search_space) > 0:
            best_func_idx = func_values.index(best_func)
            new_individual_idx = random.randint(0, len(self.search_space) - 1)
            best_func_idx = max(best_func_idx, new_individual_idx)
            self.search_space[new_individual_idx] = self.search_space[new_individual_idx][best_func_idx]

        return best_func

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func, population_size=100, mutation_prob=0.01):
        # Initialize the population
        population = [Metaheuristic(self.budget, dim) for _ in range(population_size)]

        # Run the evolution
        for _ in range(100):
            for individual in population:
                best_func = individual
                fitness = func(best_func)
                for other_individual in population:
                    if other_individual!= individual:
                        other_func = other_individual.f
                        fitness = min(fitness, func(other_func))
                individual.fitness = fitness

                # Select the best individual
                if len(individual.search_space) > 0:
                    best_func_idx = func_values.index(best_func)
                    individual.search_space = [x for x in individual.search_space if x not in best_func]
                    individual.search_space[best_func_idx] = best_func

                # Refine the strategy
                if len(individual.search_space) > 0:
                    best_func_idx = func_values.index(best_func)
                    new_individual_idx = random.randint(0, len(individual.search_space) - 1)
                    best_func_idx = max(best_func_idx, new_individual_idx)
                    individual.search_space[new_individual_idx] = individual.search_space[new_individual_idx][best_func_idx]

        # Return the best individual
        best_individual = max(population, key=lambda individual: individual.fitness)
        return best_individual

def func(func, individual, logger):
    return func(individual)

def bbob(func, population_size=100, mutation_prob=0.01):
    return EvolutionaryAlgorithm(population_size, func.__code__.co_varnames[1])

# Example usage
if __name__ == "__main__":
    func = lambda x: x**2
    population_size = 100
    mutation_prob = 0.01
    best_individual = bbob(func, population_size, mutation_prob)
    print("Best individual:", best_individual)