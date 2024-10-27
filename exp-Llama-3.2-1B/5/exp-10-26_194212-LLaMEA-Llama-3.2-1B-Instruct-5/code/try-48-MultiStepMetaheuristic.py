import random
import numpy as np
from scipy.optimize import minimize

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

    def _bounded_minimize(self, func, initial_point, initial_guess, iterations=100):
        # Perform a bounded minimization of the function
        result = minimize(func, initial_guess, method="SLSQP", bounds=self.boundaries)
        return result.x

    def func_bounded(self, point):
        # Evaluate the black box function at the given point with a bound
        return np.mean(np.square(point - np.array([0, 0, 0])))

    def func_unbounded(self, point):
        # Evaluate the black box function at the given point without bounds
        return np.mean(np.square(point - np.array([0, 0, 0])))

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
initial_point = np.array([0, 0, 0])
initial_guess = np.array([0, 0, 0])

# Optimize the function 1
bounds = self.boundaries
result1 = metaheuristic._bounded_minimize(func1, initial_point, initial_guess, iterations=100)
print("Optimized point 1:", result1.x)

# Optimize the function 2
bounds = self.boundaries
result2 = metaheuristic._bounded_minimize(func2, initial_point, initial_guess, iterations=100)
print("Optimized point 2:", result2.x)

# Optimize the function 1 without bounds
bounds = None
result3 = metaheuristic.func_bounded(func1, initial_point, initial_guess)
print("Optimized point 3:", result3)

# Optimize the function 2 without bounds
bounds = None
result4 = metaheuristic.func_unbounded(func2, initial_point, initial_guess)
print("Optimized point 4:", result4)