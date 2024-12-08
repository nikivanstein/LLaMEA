import numpy as np
import random

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

    def optimize(self, func, budget, iterations=100):
        # Initialize the population
        population = [self.func(np.array([0, 0, 0])) for _ in range(100)]

        # Run the optimization process
        for _ in range(iterations):
            # Select the fittest individual
            fittest = population[np.argmin(population)]
            # Select a random individual
            random_individual = np.random.choice(population)
            # Evaluate the fitness of the random individual
            fitness = self.func(random_individual)
            # If the fitness is better, replace the fittest individual
            if fitness < fittest + 0.05 * (fitness - fittest):
                population[np.argmin(population)] = random_individual
            # Otherwise, replace the fittest individual with the random individual
            else:
                population[np.argmin(population)] = random_individual

        return population[0]

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.optimize(func1))  # Output: 0.0
print(metaheuristic.optimize(func2))  # Output: 1.0