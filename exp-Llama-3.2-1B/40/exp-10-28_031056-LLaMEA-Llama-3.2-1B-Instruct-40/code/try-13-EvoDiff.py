# Description: EvoDiff with Refined Strategy
# Code: 
# ```python
import numpy as np

class EvoDiff:
    def __init__(self, budget, dim, population_size=100, mutation_rate=0.01):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = self.init_population()

    def init_population(self):
        # Initialize the population with random solutions
        return np.random.uniform(-5.0, 5.0, self.dim) + np.random.normal(0, 1, self.dim)

    def __call__(self, func):
        # Evaluate the black box function with the current population
        func_values = np.array([func(x) for x in self.population])

        # Select the fittest solutions
        fittest_indices = np.argsort(func_values)[::-1][:self.population_size]

        # Evolve the population using evolutionary differential evolution
        for _ in range(self.budget):
            # Select parents using tournament selection
            parents = np.array([self.population[i] for i in fittest_indices])

            # Perform mutation
            mutated_parents = parents.copy()
            for _ in range(self.population_size):
                if np.random.rand() < self.mutation_rate:
                    mutated_parents[_] += np.random.normal(0, 1, self.dim)

            # Select offspring using tournament selection
            offspring = np.array([self.population[i] for i in np.argsort(mutated_parents)[::-1][:self.population_size]])

            # Replace the old population with the new one
            self.population = np.concatenate((self.population, mutated_parents), axis=0)
            self.population = np.concatenate((self.population, offspring), axis=0)

        # Evaluate the function with the final population
        func_values = np.array([func(x) for x in self.population])
        return func_values

    def evolve(self, func, budget, dim):
        # Create a copy of the population
        new_population = self.population.copy()

        # Define the number of generations
        num_generations = 100

        # Define the probability of changing the individual lines
        prob_change = 0.4

        # Run the evolutionary differential evolution algorithm for the specified number of generations
        for _ in range(num_generations):
            # Initialize the best individual
            best_individual = new_population[0]

            # Iterate over the individuals in the population
            for i in range(len(new_population)):
                # Evaluate the function of the current individual
                func_value = func(new_population[i])

                # If the function value is better than the current best individual, update the best individual
                if func_value < best_individual[func_value.index(min(func_values))]:
                    best_individual = new_population[i]

            # Evaluate the function with the best individual
            best_individual_values = np.array([func(best_individual[i]) for i in range(self.dim)])

            # Select parents using tournament selection
            parents = np.array([new_population[i] for i in np.argsort(best_individual_values)[::-1][:self.population_size]])

            # Perform mutation
            mutated_parents = parents.copy()
            for _ in range(self.population_size):
                if np.random.rand() < prob_change:
                    mutated_parents[_] += np.random.normal(0, 1, self.dim)

            # Select offspring using tournament selection
            offspring = np.array([new_population[i] for i in np.argsort(mutated_parents)[::-1][:self.population_size]])

            # Replace the old population with the new one
            new_population = np.concatenate((new_population, mutated_parents), axis=0)
            new_population = np.concatenate((new_population, offspring), axis=0)

        return new_population