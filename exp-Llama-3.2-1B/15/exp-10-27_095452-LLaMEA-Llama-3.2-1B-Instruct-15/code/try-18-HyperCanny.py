import numpy as np
import random

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

    def select_strategy(self, func, budget, dim):
        if np.random.rand() < 0.15:
            return random_search(func, budget, dim)
        elif np.random.rand() < 0.15:
            return grid_search(func, budget, dim)
        else:
            return evolutionary_algorithm(func, budget, dim)

    def grid_search(self, func, budget, dim):
        x_values = np.linspace(-5.0, 5.0, 100)
        y_values = func(x_values)
        grid = dict(zip(x_values, y_values))
        best_x, best_y = None, None
        for x, y in grid.items():
            if x < best_x or (x == best_x and y < best_y):
                best_x, best_y = x, y
        return best_x, best_y

    def random_search(self, func, budget, dim):
        x_values = np.random.uniform(-5.0, 5.0, dim)
        y_values = func(x_values)
        return x_values, y_values

    def evolutionary_algorithm(self, func, budget, dim):
        population_size = 100
        mutation_rate = 0.1
        population = []
        for _ in range(budget):
            population.append(self.select_strategy(func, budget, dim))
        while len(population) > 0:
            new_population = []
            for _ in range(population_size):
                parent1, parent2 = random.sample(population, 2)
                child1, child2 = self.evaluate_fitness(parent1, func), self.evaluate_fitness(parent2, func)
                if np.random.rand() < 0.5:
                    child1, child2 = self.mutate(child1, child2, mutation_rate)
                new_population.append(child1, child2)
            population = new_population
        return self.select_strategy(func, budget, dim)

    def mutate(self, individual, individual, mutation_rate):
        x_values = individual[:]
        y_values = individual[1:]
        for _ in range(100):
            x_new = x_values + np.random.uniform(-0.1, 0.1, self.dim)
            y_new = y_values + np.random.uniform(-0.1, 0.1, self.dim)
            if -5.0 <= x_new <= 5.0 and -5.0 <= y_new <= 5.0:
                x_values = x_new
                y_values = y_new
                break
        return x_values, y_values

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.