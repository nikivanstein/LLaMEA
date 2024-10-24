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
        self.population = [self.x.copy() for _ in range(self.population_size)]
        self.probability = 0.05

    def __call__(self, func):
        for _ in range(self.budget):
            # Compute gradient of the objective function
            gradient = np.zeros(self.dim)
            h = 1e-1
            for i in range(self.dim):
                gradient[i] = (func(self.x + h * np.eye(self.dim)[i]) - func(self.x - h * np.eye(self.dim)[i])) / (2 * h)

            # Update the current solution using evolutionary strategy
            for individual in self.population:
                mutation = np.random.normal(0, 0.1, size=self.dim)
                individual += mutation
                individual = np.clip(individual, -5.0, 5.0)  # Clip to bounds
                f = func(individual)
                if f < func(self.x_best):
                    self.x_best = individual.copy()
                    self.f_best = f

            # Add gradient information to the evolutionary strategy
            for individual in self.population:
                mutation = 0.1 * gradient
                individual += mutation
                individual = np.clip(individual, -5.0, 5.0)  # Clip to bounds

            # Perform selection and replacement
            new_population = []
            for _ in range(self.population_size):
                parent = random.choices(self.population, weights=[func(individual) for individual in self.population])[0]
                new_population.append(parent)
            self.population = new_population

            # Perform crossover
            for i in range(self.population_size):
                if random.random() < self.probability:
                    parent1, parent2 = random.sample(self.population, 2)
                    child = parent1 + (parent2 - parent1) * random.random()
                    child = np.clip(child, -5.0, 5.0)  # Clip to bounds
                    new_population[i] = child

            # Perform mutation
            for i in range(self.population_size):
                if random.random() < self.probability:
                    mutation = np.random.normal(0, 0.1, size=self.dim)
                    new_population[i] += mutation
                    new_population[i] = np.clip(new_population[i], -5.0, 5.0)  # Clip to bounds

            # Check for convergence
            if _ % 100 == 0 and np.all(np.abs(self.x - self.x_best) < 1e-6):
                print("Converged after {} iterations".format(_))