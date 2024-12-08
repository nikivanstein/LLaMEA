import random
import numpy as np
from scipy.optimize import minimize

class MetaheuristicOptimizer:
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
        # Initialize the population with random points
        population = [self.__call__(func, iterations) for _ in range(100)]

        # Select the fittest individuals to refine the strategy
        fittest_individuals = sorted(population, key=self.func, reverse=True)[:self.budget]

        # Perform simulated annealing to refine the strategy
        while len(fittest_individuals) > 0:
            # Get the fittest individual
            fittest_individual = fittest_individuals.pop(0)

            # Initialize the new point with the fittest individual
            new_point = fittest_individual

            # Generate a new point using the current point and boundaries
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

            # Evaluate the function at the new point
            func_value = self.func(new_point)

            # If the new point is better, accept it
            if func_value > fittest_individual[func_value] * temperature:
                new_point = new_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / self.budget
                if random.random() < probability:
                    new_point = new_point
            # Update the fittest individual
            fittest_individuals.append(new_point)

        # Return the fittest individual as the optimized solution
        return fittest_individuals[0]

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

optimizer = MetaheuristicOptimizer(1000, 10)
optimized_solution = optimizer.optimize(func1, 1000)
print(optimized_solution)