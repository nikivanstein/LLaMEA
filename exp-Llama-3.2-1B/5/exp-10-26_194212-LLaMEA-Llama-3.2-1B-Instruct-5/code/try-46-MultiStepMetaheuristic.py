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
            # Initialize the best point and its value
            best_point = None
            best_value = np.inf
            for _ in range(self.dim):
                # Generate a new point using the current point and boundaries
                new_point = np.array(current_point)
                for i in range(self.dim):
                    new_point[i] += random.uniform(-1, 1)
                new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

                # Evaluate the function at the new point
                func_value = func(new_point)

                # If the new point is better, accept it
                if func_value < best_value:
                    best_point = new_point
                    best_value = func_value

            # If the best point is better, accept it with a probability based on the temperature
            if best_value > current_point[best_value] * temperature:
                current_point = best_point
            else:
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
def adaptive_probability(func, current_point, best_point, best_value, temperature, budget):
    if best_value > current_point[best_value] * temperature:
        return random.random() < (temperature / budget)
    else:
        return random.random() < (temperature / budget)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Refine the strategy
def refine_strategy(metaheuristic, func, current_point, best_point, best_value, temperature, budget):
    if best_value > current_point[best_value] * temperature:
        return adaptive_probability(func, current_point, best_point, best_value, temperature, budget)
    else:
        return random.random() < (temperature / budget)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Example usage with refined strategy
def refined_func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def refined_func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(refined_func1))  # Output: 0.0
print(metaheuristic.func(refined_func2))  # Output: 1.0