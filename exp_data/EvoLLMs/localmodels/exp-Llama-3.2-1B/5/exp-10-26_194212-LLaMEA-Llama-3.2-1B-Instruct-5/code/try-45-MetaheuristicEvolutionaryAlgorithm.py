import random
import numpy as np

class MetaheuristicEvolutionaryAlgorithm:
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

    def evolve(self, func, iterations, budget):
        # Evolve the algorithm using a population of individuals
        population = [self.func(np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)]) for _ in range(100)])
        for _ in range(iterations):
            # Select the fittest individual
            fittest_individual = population[np.argmax([self.func(individual) for individual in population])]

            # Generate a new individual using the fittest individual
            new_individual = fittest_individual
            for i in range(self.dim):
                new_individual[i] += random.uniform(-1, 1)
            new_individual = np.clip(new_individual, self.boundaries[i], self.boundaries[i+1])

            # Evaluate the new individual
            new_func_value = self.func(new_individual)

            # If the new individual is better, accept it
            if new_func_value > fittest_individual[func(new_individual)] * temperature:
                fittest_individual = new_individual
            # Otherwise, accept it with a probability based on the temperature
            else:
                probability = temperature / budget
                if random.random() < probability:
                    fittest_individual = new_individual
        return fittest_individual

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

algorithm = MetaheuristicEvolutionaryAlgorithm(1000, 10)
print(algorithm.evolve(func1, 100, 1000))
print(algorithm.evolve(func2, 100, 1000))