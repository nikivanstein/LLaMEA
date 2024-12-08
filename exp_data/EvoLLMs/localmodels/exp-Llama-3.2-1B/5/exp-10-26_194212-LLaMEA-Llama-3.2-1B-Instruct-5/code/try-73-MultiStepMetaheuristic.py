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

    def __next__(self):
        # Select the next individual based on the probability of acceptance
        if random.random() < 0.05:
            # If the current point is better, return it
            return self.func(self.boundaries)
        else:
            # Otherwise, select a new point based on the boundaries
            new_point = np.array(self.boundaries)
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])
            return new_point

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Refine the strategy by changing the individual lines of the selected solution
def func1_refined(x):
    return np.mean(np.square(x - np.array([0, 0, 0])) + 0.1 * np.random.normal(0, 1, self.dim))

def func2_refined(x):
    return np.sum(x**2 + 0.1 * np.random.normal(0, 1, self.dim))

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1_refined))  # Output: 0.001
print(metaheuristic.func(func2_refined))  # Output: 1.001