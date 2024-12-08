import random
import numpy as np
from scipy.optimize import differential_evolution

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = (-5.0, 5.0)
        self.func_evaluations = 0
        self.population_size = 100
        self.mutation_rate = 0.1

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            # Generate a population of random individuals
            population = self.generate_population(self.population_size)
            # Evaluate the function at each individual in the population
            fitness_values = [func(individual) for individual in population]
            # Select the fittest individuals
            fittest_individuals = sorted(zip(fitness_values, population), key=lambda x: x[0], reverse=True)[:self.population_size // 2]
            # Create a new generation by applying mutation
            new_generation = [individual for individual, fitness in fittest_individuals]
            for _ in range(self.population_size // 2):
                # Select two parents from the new generation
                parent1, parent2 = random.sample(new_generation, 2)
                # Calculate the mutation rate
                mutation_rate = random.random()
                # Apply mutation
                if random.random() < mutation_rate:
                    # Swap the two parents
                    new_generation.append([max(min(parent1[i], parent2[i]), -self.search_space[i]), max(min(parent1[i], parent2[i]), self.search_space[i])])
            # Replace the old population with the new generation
            population = new_generation
            # Update the best individual
            self.search_space[0], self.search_space[1] = min(self.search_space[0], min(population[0][0], population[0][1])), max(self.search_space[1], max(population[0][0], population[0][1]))
            # Increment the function evaluations
            self.func_evaluations += 1
            # Check if the budget is reached
            if self.func_evaluations >= self.budget:
                # If so, return the best individual
                return min(population, key=lambda x: x[0])[0], min(population, key=lambda x: x[0])[1]
        # If the budget is not reached, return the best individual found so far
        return self.search_space[0], self.search_space[1]

    def generate_population(self, size):
        # Generate a population of random individuals
        return [np.random.uniform(self.search_space[0], self.search_space[1], size=self.dim) for _ in range(size)]

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 