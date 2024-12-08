import numpy as np
import random
from scipy.optimize import differential_evolution

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

    def select_solution(self, func, budget, dim):
        # Initialize the population with random solutions
        population = []
        for _ in range(100):
            individual = np.random.uniform(-5.0, 5.0, self.dim)
            population.append(individual)

        # Run the evolutionary algorithm
        best_individual = None
        best_fitness = float('-inf')
        for _ in range(1000):
            # Evaluate the fitness of each individual
            fitnesses = [self.func(individual) for individual in population]
            # Select the fittest individuals
            selected_individuals = np.argsort(fitnesses)[-self.budget:]
            # Create a new population by mutating the selected individuals
            new_population = []
            for _ in range(self.budget):
                new_individual = self.func(selected_individuals[np.random.choice(selected_individuals.shape[0])])
                new_population.append(new_individual)
            # Replace the old population with the new one
            population = new_population

            # Check if the new population is better
            fitnesses = [self.func(individual) for individual in population]
            if np.max(fitnesses) > best_fitness:
                best_individual = selected_individuals[np.argsort(fitnesses)[-1]]
                best_fitness = np.max(fitnesses)

        # Refine the strategy by changing the probability of mutation
        if best_fitness / self.func_evals > 0.15:
            best_individual = self.select_solution(func, self.budget, self.dim)

        # Return the best solution
        return best_individual, best_fitness