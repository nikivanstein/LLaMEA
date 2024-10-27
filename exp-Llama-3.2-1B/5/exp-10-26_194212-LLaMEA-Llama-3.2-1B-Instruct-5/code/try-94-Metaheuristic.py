# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np
from scipy.optimize import minimize

class Metaheuristic:
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
        for _ in range(iterations):
            # Generate a new point using the current point and boundaries
            new_point = np.array(current_point)
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

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

    def func(self, point):
        # Evaluate the black box function at the given point
        return np.mean(np.square(point - np.array([0, 0, 0])))

    def optimize(self, func, iterations=100, budget=1000):
        # Define the initial population
        population = [self.__call__(func, iterations) for _ in range(50)]

        # Define the mutation function
        def mutate(individual):
            # Generate a new point by adding a random noise to the current point
            new_point = individual + np.random.normal(0, 1, self.dim)
            return new_point

        # Define the selection function
        def select(population, budget):
            # Select the best individuals based on the budget
            selected_population = []
            for _ in range(int(budget / len(population))):
                min_value = float('inf')
                min_index = -1
                for i, individual in enumerate(population):
                    value = self.func(individual)
                    if value < min_value:
                        min_value = value
                        min_index = i
                selected_population.append(population[min_index])
            return selected_population

        # Define the crossover function
        def crossover(parent1, parent2):
            # Perform crossover between two parents
            child1 = parent1[:self.dim // 2] + parent2[self.dim // 2:]
            child2 = parent2[:self.dim // 2] + parent1[self.dim // 2:]
            return child1, child2

        # Define the mutation rate
        mutation_rate = 0.05

        # Initialize the new population
        new_population = []

        # Iterate over the population
        for _ in range(iterations):
            # Select the best individuals
            population = select(population, budget)

            # Initialize the new population
            new_population = []

            # Iterate over the selected individuals
            for _ in range(int(budget / len(population))):
                # Select two parents
                parent1, parent2 = random.sample(population, 2)

                # Perform crossover
                child1, child2 = crossover(parent1, parent2)

                # Perform mutation
                child1 = mutate(child1)
                child2 = mutate(child2)

                # Add the child to the new population
                new_population.append(child1)
                new_population.append(child2)

            # Replace the old population with the new population
            population = new_population

        # Evaluate the new population
        new_fitness_values = [self.func(individual) for individual in population]

        # Get the best individual
        best_individual = population[np.argmax(new_fitness_values)]

        # Return the best individual
        return best_individual

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

best_individual = Metaheuristic(1000, 10).optimize(func1)
print(best_individual)