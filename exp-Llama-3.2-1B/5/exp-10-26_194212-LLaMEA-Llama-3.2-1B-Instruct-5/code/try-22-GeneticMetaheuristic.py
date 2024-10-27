import random
import numpy as np

class GeneticMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def __call__(self, func, iterations=100):
        # Initialize the current point and temperature
        current_point = None
        temperature = 1.0
        population_size = 100
        mutation_rate = 0.01
        selection_rate = 0.05
        for _ in range(iterations):
            # Generate a new population of size population_size
            population = self.generate_population(population_size)

            # Evaluate the function at each individual in the population
            fitness = [self.func(individual) for individual in population]

            # Select the fittest individuals
            fittest_individuals = self.select_fittest(population, fitness, selection_rate)

            # Create a new population by mutating the fittest individuals
            new_population = self.mutate(fittest_individuals, mutation_rate)

            # Evaluate the function at each individual in the new population
            fitness = [self.func(individual) for individual in new_population]

            # Replace the old population with the new one
            population = new_population

            # Update the current point
            current_point = self.update_point(population, func, boundaries)

            # If the new point is better, accept it
            if self.func(current_point) > current_point[fitness.index(max(fitness))] * temperature:
                current_point = current_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / self.budget
                if random.random() < probability:
                    current_point = current_point

        return current_point

    def generate_population(self, population_size):
        # Generate a population of size population_size
        population = np.random.uniform(self.boundaries, self.boundaries + 1, size=population_size)
        return population

    def select_fittest(self, population, fitness, selection_rate):
        # Select the fittest individuals
        fittest_individuals = np.argsort(fitness)[::-1][:selection_rate * population_size]
        return fittest_individuals

    def mutate(self, fittest_individuals, mutation_rate):
        # Mutate the fittest individuals
        mutated_individuals = []
        for individual in fittest_individuals:
            for i in range(self.dim):
                if random.random() < mutation_rate:
                    individual[i] += random.uniform(-1, 1)
            mutated_individuals.append(individual)
        return mutated_individuals

    def update_point(self, population, func, boundaries):
        # Update the current point
        current_point = None
        temperature = 1.0
        for _ in range(100):
            # Generate a new point using the current point and boundaries
            new_point = np.array(current_point)
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, boundaries[i], boundaries[i+1])

            # Evaluate the function at the new point
            func_value = func(new_point)

            # If the new point is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = new_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / self.budget
                if random.random() < probability:
                    current_point = new_point
        return current_point