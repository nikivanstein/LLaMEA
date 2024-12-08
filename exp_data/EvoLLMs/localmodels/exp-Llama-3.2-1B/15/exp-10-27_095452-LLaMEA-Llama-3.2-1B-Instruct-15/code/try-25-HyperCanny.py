import numpy as np
import random
import math

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = []

    def __call__(self, func):
        while self.func_evals < self.budget:
            # Grid search
            x_values = np.linspace(-5.0, 5.0, 100)
            y_values = func(x_values)
            grid = dict(zip(x_values, y_values))
            best_x, best_y = None, None
            for x, y in grid.items():
                if x < best_x or (x == best_x and y < best_y):
                    best_x, best_y = x, y
            # Random search
            random_x_values = np.random.uniform(-5.0, 5.0, self.dim)
            random_y_values = func(random_x_values)
            random_x_values = np.array([x for x, y in zip(random_x_values, random_y_values) if -5.0 <= x <= 5.0])
            random_y_values = np.array([y for x, y in zip(random_x_values, random_y_values) if -5.0 <= y <= 5.0])
            # Evolutionary algorithm
            self.func_evals += 1
            x_values = random_x_values
            y_values = random_y_values
            for _ in range(100):
                x_new = x_values + np.random.uniform(-0.1, 0.1, self.dim)
                y_new = y_values + np.random.uniform(-0.1, 0.1, self.dim)
                if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                    x_values = x_new
                    y_values = y_new
                    break
            # Check if the new solution is better
            if np.max(y_values) > np.max(y_values + 0.1):
                best_x, best_y = x_values, y_values
        return best_x, best_y

    def mutate(self, individual):
        if random.random() < 0.15:
            x_values = individual + np.random.uniform(-0.1, 0.1, self.dim)
            y_values = individual + np.random.uniform(-0.1, 0.1, self.dim)
            if -5.0 <= x_values <= 5.0 and -5.0 <= y_values <= 5.0:
                individual = x_values
        return individual

    def crossover(self, parent1, parent2):
        if random.random() < 0.15:
            x_values = parent1 + np.random.uniform(-0.1, 0.1, self.dim)
            y_values = parent2 + np.random.uniform(-0.1, 0.1, self.dim)
            if -5.0 <= x_values <= 5.0 and -5.0 <= y_values <= 5.0:
                parent1 = x_values
                parent2 = y_values
        return parent1, parent2

    def evaluate_fitness(self, individual):
        func_values = func(individual)
        return np.max(func_values)

    def select(self, population):
        if random.random() < 0.15:
            return random.choice(population)
        else:
            return np.random.choice(len(population), p=[0.15, 0.85])

    def run(self, func):
        population = [self.select(self.population)]
        while len(population) < 100:
            new_individual = population[-1]
            new_individual = self.evaluate_fitness(new_individual)
            new_individual = self.mutate(new_individual)
            new_individual = self.crossover(population[-1], new_individual)
            population.append(new_individual)
            self.func_evals += 1
        return population[-1]

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms with a probability of 0.15 to refine the strategy.