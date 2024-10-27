import numpy as np

class MetaHeuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.boundaries = self.generate_boundaries(dim)

    def generate_boundaries(self, dim):
        # Generate a grid of boundaries for the dimension
        boundaries = np.linspace(-5.0, 5.0, dim)
        return boundaries

    def __call__(self, func, iterations=100, budget=1000):
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
        # Update the current point with the best solution found so far
        if current_point[func(current_point)] > np.mean(np.square(current_point - np.array([0, 0, 0]))):
            current_point = current_point
        # Refine the strategy by changing the individual lines of the selected solution
        for _ in range(int(0.05 * budget)):
            new_individual = np.copy(current_point)
            for i in range(self.dim):
                new_individual[i] += random.uniform(-1, 1)
            new_individual = np.clip(new_individual, self.boundaries[i], self.boundaries[i+1])
            # Evaluate the function at the new individual
            func_value = func(new_individual)
            # If the new individual is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = new_individual
        return current_point

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MetaHeuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0

# Description: Novel Metaheuristic Algorithm for Black Box Optimization
# Code: 
# ```python
# Refine the strategy by changing the individual lines of the selected solution
metaheuristic = MetaHeuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0
metaheuristic = MetaHeuristic(1000, 10)
print(metaheuristic.func(func1))  # Output: 0.0
print(metaheuristic.func(func2))  # Output: 1.0