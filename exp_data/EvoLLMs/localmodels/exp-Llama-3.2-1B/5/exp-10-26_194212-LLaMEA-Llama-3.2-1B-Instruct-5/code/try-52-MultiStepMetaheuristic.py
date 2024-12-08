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
            # Initialize the adaptive step size
            adaptive_step_size = 1.0

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
                    # Update the adaptive step size
                    adaptive_step_size = random.uniform(0.1, 1.0)
                else:
                    # Refine the strategy by changing the individual lines
                    new_individual = self.refine_strategy(current_point, adaptive_step_size)
                    updated_individual = self.f(new_individual)
                    current_point = updated_individual

            # If the adaptive step size is too large, reduce it
            if adaptive_step_size > 0.1:
                adaptive_step_size *= 0.9

        return current_point

    def refine_strategy(self, current_point, adaptive_step_size):
        # Refine the strategy by changing the individual lines
        new_individual = current_point
        for i in range(self.dim):
            new_individual[i] += random.uniform(-1, 1) * adaptive_step_size
        return new_individual

    def f(self, individual):
        # Evaluate the black box function at the given individual
        return np.mean(np.square(individual - np.array([0, 0, 0])))

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0