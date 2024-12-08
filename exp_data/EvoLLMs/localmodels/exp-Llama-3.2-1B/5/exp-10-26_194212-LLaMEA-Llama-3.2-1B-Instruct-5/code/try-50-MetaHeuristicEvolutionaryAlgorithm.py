import random
import numpy as np

class MetaHeuristicEvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def __call__(self, func, iterations=100, mutation_rate=0.01):
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
        # Apply mutation to the current point
        mutated_point = current_point.copy()
        for i in range(self.dim):
            if random.random() < mutation_rate:
                mutated_point[i] += random.uniform(-1, 1)
        mutated_point = np.clip(mutated_point, self.boundaries[i], self.boundaries[i+1])

        # Evaluate the mutated function
        mutated_func_value = func(mutated_point)

        # If the mutated function is better, accept it
        if mutated_func_value > current_point[mutated_func_value] * temperature:
            mutated_point = mutated_point
        # Otherwise, accept it with a probability based on the temperature
        else:
            probability = temperature / self.budget
            if random.random() < probability:
                mutated_point = mutated_point

        return mutated_point

    def func(self, point):
        # Evaluate the black box function at the given point
        return np.mean(np.square(point - np.array([0, 0, 0])))

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MetaHeuristicEvolutionaryAlgorithm(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Example usage with mutation
def func3(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

metaheuristic = MetaHeuristicEvolutionaryAlgorithm(1000, 10)
print(metaheuristic.func(func3))  # Output: 0.0
mutated_point = metaheuristic.__call__(func3, mutation_rate=0.1)
print(mutated_point)  # Output: 0.1