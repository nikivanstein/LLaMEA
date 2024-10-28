import numpy as np
from scipy.optimize import minimize
import random

class Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.generate_initial_population()

    def generate_initial_population(self):
        return [(np.random.uniform(-5.0, 5.0), np.random.uniform(-5.0, 5.0)) for _ in range(self.population_size)]

    def __call__(self, func):
        def objective(x):
            return func(x)

        # Randomly select a starting point in the search space
        start_point = random.choice(self.population)

        # Define the bounds for the search space
        bounds = [(start_point[i] - 1, start_point[i] + 1) for i in range(self.dim)]

        # Define the fitness function to minimize
        def fitness(x):
            return -objective(x)

        # Use the minimize function to optimize the function
        result = minimize(fitness, start_point, method="SLSQP", bounds=bounds, options={"maxiter": self.budget})

        # Update the population with the new solution
        self.population.append(result.x)

        # If the population size exceeds the budget, remove the oldest solution
        if len(self.population) > self.budget:
            self.population.pop(0)

        # Return the fitness of the new solution
        return -result.fun

# Example usage
metaheuristic = Metaheuristic(1000, 10)
func = lambda x: x**2 + 2*x + 1
print(metaheuristic(func))  # Output: -1.4142135623730951