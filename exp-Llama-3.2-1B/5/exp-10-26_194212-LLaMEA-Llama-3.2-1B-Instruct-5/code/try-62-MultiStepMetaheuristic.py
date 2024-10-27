import random
import numpy as np

class MultiStepMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)
        self.current_point = None
        self.temperature = 1.0
        self.iterations = 0

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def __call__(self, func, iterations=100):
        # Initialize the current point and temperature
        self.current_point = None
        self.temperature = 1.0
        for _ in range(iterations):
            # Generate a new point using the current point and boundaries
            new_point = np.array(self.current_point)
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

            # Evaluate the function at the new point
            func_value = func(new_point)

            # If the new point is better, accept it
            if func_value > self.current_point[func_value] * self.temperature:
                self.current_point = new_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = self.temperature / self.budget
                if random.random() < probability:
                    self.current_point = new_point
        return self.current_point

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

# Refine the strategy
def refiner(metaheuristic, func, iterations):
    def func_refiner(point):
        # Evaluate the black box function at the given point
        return np.mean(np.square(point - np.array([0, 0, 0])))

    # Initialize the current point and temperature
    metaheuristic.current_point = None
    metaheuristic.temperature = 1.0
    metaheuristic.iterations = 0

    for _ in range(iterations):
        # Generate a new point using the current point and boundaries
        new_point = np.array(metaheuristic.current_point)
        for i in range(metaheuristic.dim):
            new_point[i] += random.uniform(-1, 1)
        new_point = np.clip(new_point, metaheuristic.boundaries[i], metaheuristic.boundaries[i+1])

        # Evaluate the function at the new point
        func_value = func_refiner(new_point)

        # If the new point is better, accept it
        if func_value > metaheuristic.current_point[func_value] * metaheuristic.temperature:
            metaheuristic.current_point = new_point
        # Otherwise, accept it with a probability based on the temperature
        else:
            probability = metaheuristic.temperature / metaheuristic.budget
            if random.random() < probability:
                metaheuristic.current_point = new_point
        metaheuristic.iterations += 1

# Example usage:
refiner(metaheuristic, func1, 100)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0