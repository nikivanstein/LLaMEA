import numpy as np
from scipy.optimize import differential_evolution

class EvolutionaryGradientDescent:
    def __init__(self, budget, dim, learning_rate, adaptive_step_size, tolerance):
        self.budget = budget
        self.dim = dim
        self.learning_rate = learning_rate
        self.adaptive_step_size = adaptive_step_size
        self.tolerance = tolerance
        self.population_size = 50
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))

    def __call__(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        for _ in range(self.budget):
            for i in range(self.population_size):
                x = self.population[i]
                fitness = objective(x)
                if fitness < self.fitnesses[i, x] + 1e-6:
                    self.fitnesses[i, x] = fitness
                    self.population[i] = x

        return self.fitnesses

    def update_solution(self, func):
        def objective(x):
            return func(x)

        def bounds(x):
            return (x.min() - 5.0, x.max() + 5.0)

        def gradient(x):
            return np.gradient(objective(x))

        def gradient_descent(x):
            return self.learning_rate * np.sign(gradient(x))

        def adaptive_step_size(x, step_size):
            if np.abs(gradient(x)) < self.adaptive_step_size:
                return np.sign(gradient(x))
            else:
                return self.adaptive_step_size

        def update(x):
            step_size = adaptive_step_size(x, self.adaptive_step_size)
            new_individual = x + step_size * gradient_descent(x)
            return new_individual

        return update

# Example usage:
budget = 1000
dim = 2
learning_rate = 0.1
adaptive_step_size = 0.01
tolerance = 1e-6

egd = EvolutionaryGradientDescent(budget, dim, learning_rate, adaptive_step_size, tolerance)
egd.update_solution(func)

# Print the updated solution
updated_individual = egd.population[0]
print(updated_individual)