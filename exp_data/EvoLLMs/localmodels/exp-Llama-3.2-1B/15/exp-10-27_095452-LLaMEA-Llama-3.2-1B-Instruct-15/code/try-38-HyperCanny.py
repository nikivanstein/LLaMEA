import numpy as np
import random
from scipy.optimize import minimize

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

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

    def _get_neighbor(self, individual, dim):
        x_values = individual
        y_values = np.zeros_like(x_values)
        for _ in range(100):
            x_new = x_values + np.random.uniform(-0.1, 0.1, dim)
            y_new = y_values + np.random.uniform(-0.1, 0.1, dim)
            if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                x_values = x_new
                y_values = y_new
                break
        return x_values, y_values

    def _get_neighbor_stronger(self, individual, dim):
        x_values = individual
        y_values = np.zeros_like(x_values)
        for _ in range(100):
            x_new = x_values + np.random.uniform(-0.1, 0.1, dim)
            y_new = y_values + np.random.uniform(-0.1, 0.1, dim)
            if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                x_values = x_new
                y_values = y_new
                break
        return x_values, y_values

    def _get_neighbor_finer(self, individual, dim):
        x_values = individual
        y_values = np.zeros_like(x_values)
        for _ in range(100):
            x_new = x_values + np.random.uniform(-0.1, 0.1, dim)
            y_new = y_values + np.random.uniform(-0.1, 0.1, dim)
            if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                x_values = x_new
                y_values = y_new
                break
        return x_values, y_values

    def evaluate_fitness(self, individual, logger):
        x_values, y_values = self._get_neighbor(individual, self.dim)
        new_individual = self.func_evals * x_values + y_values
        logger.update(individual, new_individual)
        return new_individual

    def update_individual(self, individual, new_individual):
        self.func_evals += 1
        return new_individual

    def _randomize(self, individual):
        return random.uniform(-5.0, 5.0), random.uniform(-5.0, 5.0)

    def _crossover(self, parent1, parent2):
        x_values1, y_values1 = parent1
        x_values2, y_values2 = parent2
        crossover_point = np.random.randint(1, len(x_values1))
        child_x_values = x_values1[:crossover_point] + x_values2[crossover_point:]
        child_y_values = y_values1[:crossover_point] + y_values2[crossover_point:]
        return child_x_values, child_y_values

    def _mutation(self, individual):
        x_values, y_values = individual
        mutation_point = np.random.randint(1, len(x_values))
        mutated_x_values = x_values[:mutation_point] + np.random.uniform(-0.1, 0.1, x_values[mutation_point:])
        mutated_y_values = y_values[:mutation_point] + np.random.uniform(-0.1, 0.1, y_values[mutation_point:])
        return mutated_x_values, mutated_y_values