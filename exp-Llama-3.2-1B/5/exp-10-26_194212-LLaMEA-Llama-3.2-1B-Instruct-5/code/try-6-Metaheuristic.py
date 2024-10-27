import random
import numpy as np

class Metaheuristic:
    def __init__(self, budget, dim, probabilities):
        self.budget = budget
        self.dim = dim
        self.probabilities = probabilities
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

            # Calculate the probability of accepting the new point
            probabilities = self.probabilities[func_value]
            acceptance_probability = min(1.0, 1 / self.budget + 0.05 * np.clip(1.0 - probabilities, 0, 1))

            # If the new point is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = new_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                current_point = new_point
                if random.random() < acceptance_probability:
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

probabilities = [0.6, 0.4]  # Probability of accepting the new point with a better function value
metaheuristic = Metaheuristic(1000, 10, probabilities)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

An excellent and novel heuristic algorithm to solve this task is:
# Description: Adaptive Local Search with Temperature Control
# Code: 