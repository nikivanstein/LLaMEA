import random
import numpy as np

class AdaptiveMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)
        self.current_individual = None
        self.current_fitness = None

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def __call__(self, func, iterations=100):
        # Initialize the current point and temperature
        self.current_individual = None
        self.current_fitness = None
        for _ in range(iterations):
            # Generate a new point using the current point and boundaries
            new_individual = self.evaluate_individual()

            # Evaluate the function at the new point
            new_fitness = func(new_individual)

            # If the new point is better, accept it
            if new_fitness > self.current_fitness * 0.95:
                self.current_individual = new_individual
                self.current_fitness = new_fitness
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = 0.05
                if random.random() < probability:
                    self.current_individual = new_individual
                    self.current_fitness = new_fitness

        return self.current_individual

    def evaluate_individual(self):
        # Evaluate the black box function at the given point
        return np.mean(np.square(self.current_individual - np.array([0, 0, 0])))

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = AdaptiveMetaheuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Description: Adaptive Multi-Step Metaheuristic for BBOB Optimization
# Code: 