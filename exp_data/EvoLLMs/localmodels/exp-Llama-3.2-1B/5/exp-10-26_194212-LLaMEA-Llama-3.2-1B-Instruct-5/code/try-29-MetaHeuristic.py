import random
import numpy as np
from collections import deque

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)
        self.population = deque(maxlen=1000)

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

    def mutate(self, func, current_point):
        # Mutate the current point by adding a random noise
        new_point = current_point.copy()
        for i in range(self.dim):
            new_point[i] += random.uniform(-1, 1)
        new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])
        return new_point

    def evaluate_fitness(self, func, current_point):
        # Evaluate the function at the current point
        func_value = func(current_point)
        return func_value

    def select(self, func, current_point, population):
        # Select the individual with the highest fitness
        fitness = self.evaluate_fitness(func, current_point)
        selected_individuals = [individual for individual in population if individual[fitness] == fitness]
        return selected_individuals[:self.budget]

    def __next__(self):
        # Select the next individual and mutate it
        selected_individuals = self.select(self.func, self.current_point, self.population)
        selected_individuals = [self.mutate(self.func, individual) for individual in selected_individuals]
        self.population = deque(selected_individuals)
        if len(self.population) >= self.budget:
            self.current_point = self.population.popleft()
        return self.current_point

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MetaHeuristic(1000, 10)
for _ in range(100):
    print(metaheuristic.__next__())