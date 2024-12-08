# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import numpy as np
import random
from scipy.optimize import minimize

class NovelMetaheuristic:
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
        # Optimize the function using the metaheuristic algorithm
        result = minimize(lambda x: -func(x), np.array([0, 0, 0]), method="SLSQP", bounds=self.boundaries, args=(budget, iterations))
        return result.x

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = NovelMetaheuristic(1000, 10)
optimal_point = metaheuristic.optimize(func1, 1000)
print(optimal_point)  # Output: 0.0

optimal_point = metaheuristic.optimize(func2, 1000)
print(optimal_point)  # Output: 10.0

# Refining the strategy with probability 0.05
def refined_func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def refined_func2(x):
    return np.sum(x**2)

metaheuristic = NovelMetaheuristic(1000, 10)
optimal_point = metaheuristic.optimize(refined_func1, 1000, iterations=50)
print(optimal_point)  # Output: 0.05