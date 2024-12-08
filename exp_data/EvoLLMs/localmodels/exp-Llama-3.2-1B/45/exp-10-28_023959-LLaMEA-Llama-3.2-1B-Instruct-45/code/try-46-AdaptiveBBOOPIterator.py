import random
import numpy as np

class AdaptiveBBOOPIterator:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = 2 * np.random.uniform(-5.0, 5.0, dim)
        self.population = [self.search_space] * dim
        self.fitness_values = np.zeros(dim)
        self.population_indices = list(range(dim))

    def __call__(self, func):
        def __call_individual(individual):
            mutated_individual = individual.copy()
            if random.random() < 0.45:
                mutated_individual[random.randint(0, self.dim - 1)] += random.uniform(-1.0, 1.0)
            return mutated_individual

        # Select parents using tournament selection
        tournament_indices = random.sample(self.population_indices, 3)
        tournament_fitness_values = np.array([self.fitness_values[i] for i in tournament_indices])
        tournament_parents = []
        for i in range(3):
            parent_index = tournament_indices[i]
            parent_fitness_value = tournament_fitness_values[i]
            parent_index = random.choice(self.population_indices)
            parent_fitness_value = self.fitness_values[parent_index]
            if parent_fitness_value < parent_fitness_value:
                parent_index = parent_index
            tournament_parents.append(self.__call_individual(self.population[parent_index]))

        # Evolve the population using mutation and selection
        for _ in range(self.budget):
            parents = tournament_parents
            # Select the best parent based on fitness
            best_parent_index = np.argmax(self.fitness_values)
            self.population[self.population_indices[best_parent_index]] = self.__call_individual(self.population[best_parent_index])
            self.fitness_values[best_parent_index] = func(self.__call_individual(self.population[best_parent_index]))

        # Return the best individual
        return self.get_best_individual()

# Description: Adaptive Black Box Optimization with Metaheuristic Evolutionary Strategies
# Code: 