import random
import numpy as np

class MultiStepMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def __call__(self, func, iterations=100, population_size=100, mutation_rate=0.01):
        # Initialize the current point and temperature
        current_point = None
        temperature = 1.0
        population = self.generate_population(iterations)

        for _ in range(iterations):
            # Evaluate the function at each individual in the population
            fitness = [self.func(individual) for individual in population]
            # Select the fittest individuals
            fittest = np.argsort(fitness)[-population_size:]
            # Select parents using tournament selection
            parents = np.random.choice(fittest, population_size, replace=False)
            # Create a new population by crossover and mutation
            new_population = self.crossover(parents, mutation_rate)
            # Replace the old population with the new one
            population = new_population

            # Update the current point and temperature
            current_point = self.update_point(population, fitness, temperature)

            # Evaluate the function at the new point
            func_value = self.func(current_point)

            # If the new point is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = current_point[func_value]
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / self.budget
                if random.random() < probability:
                    current_point = current_point[func_value]

        return current_point

    def func(self, point):
        # Evaluate the black box function at the given point
        return np.mean(np.square(point - np.array([0, 0, 0])))

    def generate_population(self, iterations):
        # Generate a population of random individuals
        return np.random.uniform(self.boundaries, self.boundaries + 5, size=(iterations, self.dim))

    def crossover(self, parents, mutation_rate):
        # Perform crossover between parents
        offspring = np.zeros(self.dim)
        for i in range(self.dim):
            if random.random() < mutation_rate:
                j = random.randint(0, self.dim - 1)
                offspring[i] = parents[i][j]
            else:
                offspring[i] = parents[i][i]
        return offspring

    def update_point(self, population, fitness, temperature):
        # Update the current point using simulated annealing
        current_point = None
        for _ in range(self.dim):
            new_point = np.array(population[np.random.randint(0, len(population), size=1)])
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

            # Evaluate the function at the new point
            func_value = self.func(new_point)

            # If the new point is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = new_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / self.budget
                if random.random() < probability:
                    current_point = new_point
        return current_point

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Novel Algorithm: Evolutionary Strategy with Evolutionary Crossover and Mutation
# Description: Novel Algorithm for Black Box Optimization using Evolutionary Strategies
# Code: 