import numpy as np
import random

class EvolutionaryGradient:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.x = np.random.uniform(-5.0, 5.0, size=dim)
        self.f_best = np.inf
        self.x_best = None
        self.population_size = 100

    def __call__(self, func):
        for _ in range(self.budget):
            # Initialize population
            population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
            f_values = np.zeros(self.population_size)

            # Evaluate fitness of population
            for i in range(self.population_size):
                f_values[i] = func(population[i])

            # Selection
            indices = np.argsort(f_values)
            selected_indices = indices[:int(self.population_size * 0.975)]
            selected_population = population[selected_indices]

            # Crossover
            offspring = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                parent1, parent2 = selected_population[np.random.randint(0, self.population_size, 2)]
                crossover_point = np.random.randint(1, self.dim)
                offspring[i] = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            # Mutation
            for i in range(self.population_size):
                mutation = np.random.normal(0, 0.1, size=self.dim)
                offspring[i] += mutation

            # Replacement
            population = np.concatenate((selected_population, offspring))

            # Update best solution
            f_values = np.zeros(self.population_size)
            for i in range(self.population_size):
                f_values[i] = func(population[i])
            if np.min(f_values) < self.f_best:
                self.f_best = np.min(f_values)
                self.x_best = population[np.argmin(f_values)]

            # Add gradient information
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)
            self.x += 0.1 * gradient

            # Check for convergence
            if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < 1e-6):
                print("Converged after {} iterations".format(_))

# Example usage:
def func(x):
    return np.sum(x**2)

evg = EvolutionaryGradient(budget=1000, dim=10)
evg("func")