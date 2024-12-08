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

class GeneticAlgorithm(Metaheuristic):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def __call__(self, func):
        population_size = 100
        mutation_rate = 0.01

        # Initialize the population
        population = [self.__call__(func) for _ in range(population_size)]

        # Evaluate the fitness of each individual
        fitnesses = [self.evaluate_fitness(individual) for individual in population]

        # Select the fittest individuals
        fittest_individuals = [individual for _, individual in sorted(zip(fitnesses, population), reverse=True)]

        # Create a new population by breeding the fittest individuals
        new_population = []
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(fittest_individuals, 2)
            child = (parent1 + parent2) / 2
            if random.random() < 0.5:
                child = parent1
            new_population.append(child)

        # Evaluate the fitness of the new population
        new_fitnesses = [self.evaluate_fitness(individual) for individual in new_population]

        # Select the fittest individuals in the new population
        fittest_individuals = [individual for _, individual in sorted(zip(new_fitnesses, new_population), reverse=True)]

        # Merge the fittest individuals into a single individual
        new_individual = fittest_individuals[0]
        for individual in fittest_individuals[1:]:
            new_individual = [x for x in new_individual if x not in individual]

        # Update the search space
        new_individual = [x for x in new_individual if x not in new_individual]

        return new_individual

# Description: Novel Metaheuristic Algorithm for Black Box Optimization using Genetic Algorithm
# Code: 