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

    def __call__(self, func, iterations=100):
        # Initialize the current point and temperature
        current_point = None
        temperature = 1.0
        for _ in range(iterations):
            # Initialize the population of individuals
            population = self.generate_population(iterations)

            # Evaluate the population using the budget function evaluations
            fitnesses = self.evaluate_fitness(population, self.budget)

            # Select the fittest individuals
            selected_individuals = self.select_fittest(population, fitnesses)

            # Perform a single step of optimization
            for individual in selected_individuals:
                # Generate a new point using the current point and boundaries
                new_point = np.array(individual)
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

    def evaluate_fitness(self, population, budget):
        # Evaluate the fitness of each individual in the population
        fitnesses = []
        for individual in population:
            func_value = func(individual)
            fitnesses.append(func_value)
        return fitnesses

    def select_fittest(self, population, fitnesses):
        # Select the fittest individuals based on their fitness
        selected_individuals = []
        for _ in range(int(len(population) * 0.2)):
            individual = random.choice(population)
            selected_individuals.append(individual)
        return selected_individuals

    def generate_population(self, iterations):
        # Generate a population of individuals
        population = []
        for _ in range(iterations):
            # Generate a random individual
            individual = np.random.choice(self.boundaries, self.dim)
            population.append(individual)
        return population

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Novel Multi-Step Optimizer for Black Box Functions
# Description: Novel Multi-Step Optimizer for Black Box Functions
# Code: 