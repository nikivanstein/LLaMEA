# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
import random
import numpy as np

class Metaheuristic:
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

    def optimize(self, func, iterations=100, budget=1000):
        # Initialize the population with random points
        population = [self.func(np.array([random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)]) for _ in range(50))]

        while len(population) < self.budget:
            # Evaluate the fitness of each individual
            fitness = [self.func(individual) for individual in population]

            # Select the fittest individuals
            fittest_indices = np.argsort(fitness)[-self.budget:]
            fittest_individuals = [population[i] for i in fittest_indices]

            # Create new individuals by refining the fittest individuals
            new_individuals = []
            for _ in range(50):
                # Select two parents from the fittest individuals
                parent1, parent2 = random.sample(fittest_individuals, 2)
                # Create a child by combining the parents
                child = np.array(parent1) + 0.5 * (parent2 - parent1)
                # Evaluate the fitness of the child
                fitness_child = self.func(child)
                # Add the child to the new individuals list
                new_individuals.append(child)

            # Add the new individuals to the population
            population.extend(new_individuals)

        # Return the fittest individual
        return population[0]

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = Metaheuristic(1000, 10)
print(metaheuristic.optimize(func1))  # Output: 0.0
print(metaheuristic.optimize(func2))  # Output: 1.0