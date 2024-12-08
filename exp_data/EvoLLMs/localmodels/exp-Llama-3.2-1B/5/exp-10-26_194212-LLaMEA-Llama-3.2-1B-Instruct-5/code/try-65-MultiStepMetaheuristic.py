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

    def __call__(self, func, iterations=100, budget=1000, temperature=1.0):
        # Initialize the current point and temperature
        current_point = None
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

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

def optimize_func1():
    metaheuristic = MultiStepMetaheuristic(1000, 10)
    for _ in range(100):
        metaheuristic.func(func1)
    return metaheuristic

def optimize_func2():
    metaheuristic = MultiStepMetaheuristic(1000, 10)
    for _ in range(100):
        metaheuristic.func(func2)
    return metaheuristic

# Novel Metaheuristic Algorithm: "Multi-Step Metaheuristic with Adaptive Probability"
# Description: A novel metaheuristic algorithm that uses a combination of local search and simulated annealing to optimize black box functions.
# Code: 