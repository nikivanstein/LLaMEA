import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)
        self.population = self.initialize_population()

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def initialize_population(self):
        # Initialize the population with random individuals
        return [self.generate_individual() for _ in range(100)]

    def generate_individual(self):
        # Generate an individual using the boundaries
        return np.array([random.uniform(boundaries[i], boundaries[i+1]) for i in range(self.dim)])

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

    def fitness(self, individual):
        # Evaluate the fitness of an individual
        return np.mean(np.square(individual - np.array([0, 0, 0])))

    def mutate(self, individual):
        # Mutate an individual
        return np.array([random.uniform(-1, 1) for _ in range(self.dim)])

    def __str__(self):
        # Return a string representation of the optimizer
        return "Black Box Optimizer with population size " + str(len(self.population)) + ", budget " + str(self.budget) + ", and dimension " + str(self.dim)

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

optimizer = BlackBoxOptimizer(1000, 10)
print(optimizer)