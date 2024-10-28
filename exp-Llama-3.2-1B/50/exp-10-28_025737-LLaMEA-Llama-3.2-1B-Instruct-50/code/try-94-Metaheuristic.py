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
        self.search_space = [x for x in self.search_space if x not in best_func]

        return best_func

class NovelMetaheuristicAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, (dim,))

    def __call__(self, func):
        # Initialize the population with random solutions
        population = [self._initialize_population(func, self.budget, self.dim) for _ in range(100)]

        # Evolve the population
        for _ in range(1000):
            # Select the fittest individuals
            fittest_individuals = sorted(population, key=lambda x: x.fitness, reverse=True)[:self.budget]

            # Create new individuals
            new_population = []
            for _ in range(self.budget):
                # Select two parents from the fittest individuals
                parent1, parent2 = random.sample(fittest_individuals, 2)

                # Create a new individual by combining the parents
                child = parent1[:self.dim // 2] + parent2[self.dim // 2:]

                # Evaluate the new individual
                child.fitness = func(child)

                # Add the new individual to the new population
                new_population.append(child)

            # Update the population
            population = new_population

        # Return the best individual
        return population[0].fitness

    def _initialize_population(self, func, budget, dim):
        # Initialize the population with random solutions
        population = [Metaheuristic(1, dim) for _ in range(budget)]

        for individual in population:
            # Evaluate the function a limited number of times
            num_evals = min(self.budget, len(func(individual.search_space)))
            func_values = [func(individual.search_space) for individual in population]
            individual.search_space = [x for x in individual.search_space if x not in func_values]

        return population

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 