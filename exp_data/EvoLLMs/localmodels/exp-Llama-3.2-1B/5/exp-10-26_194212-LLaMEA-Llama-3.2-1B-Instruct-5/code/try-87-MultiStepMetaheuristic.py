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
            # Generate a new point using the current point and boundaries
            new_point = np.array(current_point)
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

            # Evaluate the function at the new point
            func_value = func(new_point)

            # Calculate the probability of acceptance based on the budget and temperature
            probability = 1.0 / self.budget
            if random.random() < probability:
                current_point = new_point
            else:
                # Refine the strategy by changing the individual lines of the selected solution
                for j in range(self.dim):
                    if random.random() < 0.05:
                        current_point[j] += random.uniform(-0.1, 0.1)
                temperature *= 0.9

            # If the new point is better, accept it
            if func_value > current_point[func_value] * temperature:
                current_point = new_point
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

# An exception occured: Traceback (most recent call last):
#  File "/root/LLaMEA/llamea/llamea.py", line 187, in initialize_single
#    new_individual = self.evaluate_fitness(new_individual)
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "/root/LLaMEA/llamea/llamea.py", line 264, in evaluate_fitness
#    updated_individual = self.f(individual, self.logger)
#                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#     TypeError: evaluateBBOB() missing 1 required positional argument: 'budget'
#.