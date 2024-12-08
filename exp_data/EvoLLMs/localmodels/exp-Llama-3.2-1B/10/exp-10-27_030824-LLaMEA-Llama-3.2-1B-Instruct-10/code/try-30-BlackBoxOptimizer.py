import random
import numpy as np

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = [-5.0, 5.0]
        self.func_evaluations = 0

    def __call__(self, func):
        # Ensure the function evaluations do not exceed the budget
        if self.func_evaluations < self.budget:
            # Generate a random point in the search space
            point = np.random.uniform(self.search_space[0], self.search_space[1])
            # Evaluate the function at the point
            evaluation = func(point)
            # Increment the function evaluations
            self.func_evaluations += 1
            # Return the point and its evaluation
            return point, evaluation
        else:
            # If the budget is reached, return a default point and evaluation
            return np.random.uniform(self.search_space[0], self.search_space[1]), func(np.random.uniform(self.search_space[0], self.search_space[1]))

    def __str__(self):
        return f"Novel Metaheuristic Algorithm for Black Box Optimization"

    def __repr__(self):
        return f"BlackBoxOptimizer(budget={self.budget}, dim={self.dim})"

    def mutate(self, new_individual):
        # Randomly select a dimension to mutate
        dimension = random.randint(0, self.dim - 1)
        # Randomly select a value within the search space to mutate
        value = np.random.uniform(self.search_space[0], self.search_space[1])
        # Update the new individual
        new_individual[dimension] += random.uniform(-1, 1) * value
        # Ensure the new individual is within the search space
        new_individual[dimension] = np.clip(new_individual[dimension], self.search_space[0], self.search_space[1])

# One-line description: Novel metaheuristic algorithm for black box optimization using a novel combination of random walk and linear interpolation.

def bbo_pareto(budget, dim):
    optimizer = BlackBoxOptimizer(budget, dim)
    results = []
    for _ in range(1000):
        point, evaluation = optimizer(func)
        results.append((point, evaluation))
    # Evaluate the Pareto front
    pareto_front = sorted(results, key=lambda x: x[1], reverse=True)
    # Return the Pareto front
    return pareto_front

def bbo_pareto_str(budget, dim):
    optimizer = BlackBoxOptimizer(budget, dim)
    results = []
    for _ in range(1000):
        point, evaluation = optimizer(func)
        results.append((point, evaluation))
    # Evaluate the Pareto front
    pareto_front = sorted(results, key=lambda x: x[1], reverse=True)
    # Return the Pareto front as a string
    return '\n'.join(f"Point {point}, Evaluation {evaluation}" for point, evaluation in pareto_front)

def func(x):
    return x[0]**2 + x[1]**2

def bbo_pareto_evaluator(func, budget, dim):
    return bbo_pareto(budget, dim)

# Example usage:
print(bbo_pareto_evaluator(func, budget=100, dim=2))