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
            # Initialize the current point
            current_point = None
            for _ in range(self.dim):
                current_point.append(random.uniform(-1, 1))
            current_point = np.array(current_point)

            # Evaluate the function at the current point
            func_value = func(current_point)

            # Initialize the best point and its value
            best_point = None
            best_value = float('-inf')
            for _ in range(self.budget):
                # Generate a new point using the current point and boundaries
                new_point = np.array(current_point)
                for i in range(self.dim):
                    new_point[i] += random.uniform(-1, 1)
                new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

                # Evaluate the function at the new point
                func_value = func(new_point)

                # If the new point is better, update the best point and its value
                if func_value > best_value:
                    best_point = new_point
                    best_value = func_value

            # Accept the best point with a probability based on the temperature
            probability = temperature / self.budget
            if random.random() < probability:
                current_point = best_point
            else:
                # Otherwise, accept it with a probability based on the temperature
                probability = temperature / self.budget
                if random.random() < probability:
                    current_point = best_point

        return current_point

    def func(self, point):
        # Evaluate the black box function at the given point
        return np.mean(np.square(point - np.array([0, 0, 0])))

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Adaptive probability of acceptance
def func1_adaptive(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2_adaptive(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1_adaptive))  # Output: 0.0
print(metaheuristic.func(func2_adaptive))  # Output: 1.0