import numpy as np
import random

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.x_values = None
        self.y_values = None
        self.best_x = None
        self.best_y = None
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
            # Refine the solution
            if best_x is None:
                self.refine_solution(func, x_values, y_values)
            else:
                self.population.append((best_x, best_y, x_values, y_values, np.max(y_values), np.max(y_values + 0.1)))
        # Select the best individual
        self.best_individual = self.population[np.argmax([item[3] for item in self.population])][0]
        self.best_individual = self.evaluate_fitness(self.best_individual)
        return self.best_individual

    def evaluate_fitness(self, individual):
        # Grid search
        x_values = individual[3]
        y_values = individual[4]
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

    def refine_solution(self, func, x_values, y_values):
        # Refine the solution by changing the bounds
        x_values = x_values + np.random.uniform(-0.1, 0.1, self.dim)
        y_values = y_values + np.random.uniform(-0.1, 0.1, self.dim)
        if -5.0 <= x_values[0] <= 5.0 and -5.0 <= y_values[0] <= 5.0:
            x_values = x_values
            y_values = y_values
            best_x, best_y = x_values, y_values
        # Refine the bounds using the bounds from the best solution
        if best_x is not None:
            x_values = np.clip(x_values, best_x - 0.1, best_x + 0.1)
            y_values = np.clip(y_values, best_y - 0.1, best_y + 0.1)

    def select_best(self):
        # Select the best individual based on the fitness
        fitness = [item[3] for item in self.population]
        best_index = np.argmax(fitness)
        best_individual = self.population[best_index]
        return best_individual

    def update(self, budget):
        # Update the population
        self.func_evals = 0
        self.x_values = None
        self.y_values = None
        self.best_individual = self.select_best()
        self.population = []
        while self.func_evals < budget:
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
            # Refine the solution
            if best_x is None:
                self.refine_solution(func, x_values, y_values)
            else:
                self.population.append((best_x, best_y, x_values, y_values, np.max(y_values), np.max(y_values + 0.1)))
        # Select the best individual
        self.best_individual = self.select_best()
        return self.best_individual

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.