import random
import numpy as np

class AdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)
        self.iterations = 0
        self.accept_probability = 0.05

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

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = AdaptiveMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Adaptive strategy: 0.9 for local search and 0.1 for simulated annealing
def adaptive_strategy(func1, func2):
    metaheuristic.iterations += 1
    if metaheuristic.iterations >= 1000:
        return metaheuristic
    if np.random.rand() < metaheuristic.accept_probability:
        metaheuristic = AdaptiveMetaheuristic(1000, 10)
    return metaheuristic

adaptive_metaheuristic = adaptive_strategy(func1, func2)
print(adaptive_metaheuristic.func(func1))  # Output: 0.9
print(adaptive_metaheuristic.func(func2))  # Output: 1.0