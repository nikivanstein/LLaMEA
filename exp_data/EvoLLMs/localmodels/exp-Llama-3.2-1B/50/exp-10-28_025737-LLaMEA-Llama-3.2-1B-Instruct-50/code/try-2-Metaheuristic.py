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
        self.algorithm = Metaheuristic(budget, dim)
        self.population_size = 100
        self.population = self.initialize_single()

    def initialize_single(self):
        return [self.algorithm.__call__(np.random.uniform(-5.0, 5.0, self.dim)) for _ in range(self.population_size)]

    def select(self, population):
        # Select the best individual based on the probability of 0.45
        probabilities = [self.algorithm.func(individual) / len(population) for individual in population]
        selected_indices = np.random.choice(len(population), size=self.population_size, p=probabilities)
        selected_individuals = [population[i] for i in selected_indices]

        # Create a new population with the selected individuals
        new_population = self.algorithm.__call__(np.array(selected_individuals))

        return new_population

    def mutate(self, new_population):
        # Mutate each individual with a probability of 0.1
        mutated_individuals = []
        for individual in new_population:
            mutated_individuals.append(individual + random.uniform(-0.1, 0.1))
        return mutated_individuals

    def evolve(self, population, iterations):
        for _ in range(iterations):
            population = self.select(population)
            population = self.mutate(population)

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 