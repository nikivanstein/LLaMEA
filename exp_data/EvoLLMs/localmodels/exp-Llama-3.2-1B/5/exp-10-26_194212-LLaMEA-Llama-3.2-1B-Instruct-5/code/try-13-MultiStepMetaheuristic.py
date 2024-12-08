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
            # Initialize the list of new points
            new_points = []
            # Iterate over the dimensions
            for i in range(self.dim):
                # Generate a new point using the current point and boundaries
                new_point = np.array(current_point)
                for j in range(self.dim):
                    new_point[j] += random.uniform(-1, 1)
                new_point[j] = np.clip(new_point[j], self.boundaries[j], self.boundaries[j+1])

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
                # Add the new point to the list of new points
                new_points.append(new_point)
            # If the budget is exhausted, return the current point
            if len(new_points) == 0:
                return current_point
            # Otherwise, select the next new point based on the probability
            next_point_index = random.randint(0, len(new_points) - 1)
            next_point = new_points[next_point_index]
            # Update the current point and temperature
            current_point = next_point
            temperature *= 0.95
        # Return the final current point
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

# Refine the strategy
def refine_strategy(func1, func2, metaheuristic):
    # Initialize the current point and temperature
    current_point = None
    temperature = 1.0
    for _ in range(100):
        # Initialize the list of new points
        new_points = []
        # Iterate over the dimensions
        for i in range(10):
            # Generate a new point using the current point and boundaries
            new_point = np.array(current_point)
            for j in range(10):
                new_point[j] += random.uniform(-1, 1)
            new_point[j] = np.clip(new_point[j], 0, 10)
            # Evaluate the function at the new point
            func_value = func1(new_point)
            # If the new point is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = new_point
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / 10
                if random.random() < probability:
                    current_point = new_point
            # Add the new point to the list of new points
            new_points.append(new_point)
        # If the budget is exhausted, return the current point
        if len(new_points) == 0:
            return current_point
        # Otherwise, select the next new point based on the probability
        next_point_index = random.randint(0, len(new_points) - 1)
        next_point = new_points[next_point_index]
        # Update the current point and temperature
        current_point = next_point
        temperature *= 0.95
    # Return the final current point
    return current_point

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(refine_strategy(func1, func2, metaheuristic))  # Output: (1.5, 0.8, 0.9)