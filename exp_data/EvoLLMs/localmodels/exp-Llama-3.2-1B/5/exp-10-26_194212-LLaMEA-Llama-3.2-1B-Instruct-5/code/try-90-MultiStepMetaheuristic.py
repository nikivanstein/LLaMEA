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

    def optimize(self, func, iterations=100):
        # Optimize the function using the Multi-Step Metaheuristic
        best_point = None
        best_score = float('-inf')
        for _ in range(iterations):
            # Generate a new point using the current point and boundaries
            new_point = np.array(self.func(0))
            for i in range(self.dim):
                new_point[i] += random.uniform(-1, 1)
            new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])

            # Evaluate the function at the new point
            new_score = self.func(new_point)

            # If the new point is better, update the best point and score
            if new_score > best_score:
                best_point = new_point
                best_score = new_score

            # If the best point is better, accept it with a probability based on the temperature
            if best_score > best_point[best_score] * temperature:
                best_point = new_point
        return best_point

# Example usage:
def func1(x):
    return np.mean(np.square(x - np.array([0, 0, 0])))

def func2(x):
    return np.sum(x**2)

metaheuristic = MultiStepMetaheuristic(1000, 10)
best_point = metaheuristic.optimize(func1)
print(best_point)  # Output: a better point
print(metaheuristic.func(best_point))  # Output: a better score

# Refine the strategy with a probability of 0.05
def refine_strategy(point, score):
    # Randomly decide whether to refine the strategy
    if random.random() < 0.05:
        # Refine the strategy by changing the step size
        step_size = random.uniform(0.1, 0.5)
        new_point = np.array(point)
        for i in range(self.dim):
            new_point[i] += random.uniform(-step_size, step_size)
        new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])
        return new_point, score
    else:
        return point, score

def refine_strategy2(point, score):
    # Refine the strategy by changing the step size and adding noise
    step_size = random.uniform(0.1, 0.5)
    new_point = np.array(point)
    for i in range(self.dim):
        new_point[i] += random.uniform(-step_size, step_size)
    new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])
    new_point += np.random.uniform(-1, 1, self.dim)
    new_point = np.clip(new_point, self.boundaries[i], self.boundaries[i+1])
    return new_point, score

metaheuristic = MultiStepMetaheuristic(1000, 10)
best_point = metaheuristic.optimize(func1, iterations=200)
print(best_point)  # Output: a better point
print(metaheuristic.func(best_point))  # Output: a better score

# Further optimize the function using the refined strategy
best_point2 = refine_strategy(best_point, metaheuristic.func(best_point))
print(best_point2)  # Output: a better point
print(metaheuristic.func(best_point2))  # Output: a better score

# Further optimize the function using the refined strategy with a probability of 0.05
best_point3 = refine_strategy2(best_point2, metaheuristic.func(best_point2))
print(best_point3)  # Output: a better point
print(metaheuristic.func(best_point3))  # Output: a better score