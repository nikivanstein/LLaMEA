import numpy as np

class HyperCanny:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0
        self.population = None

    def __call__(self, func):
        if self.population is None:
            self.population = self.generate_initial_population(func, self.budget, self.dim)
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

    def generate_initial_population(self, func, budget, dim):
        population = []
        for _ in range(budget):
            individual = np.random.uniform(-5.0, 5.0, dim)
            population.append(individual)
        return population

    def select_next_individual(self, population):
        if len(population) == 0:
            return None
        if len(population) == 1:
            return population[0]
        fitness_values = [self.evaluate_fitness(individual, func) for individual, func in zip(population, self.func)]
        selected_index = np.argsort(fitness_values)[::-1][0:1]
        selected_individual = population[selected_index]
        return selected_individual

    def mutate(self, individual):
        if np.random.rand() < 0.15:
            return individual + np.random.uniform(-0.1, 0.1, self.dim)
        return individual

    def evaluate_fitness(self, individual, func):
        fitness_values = [func(individual) for individual in self.population]
        return np.max(fitness_values)

# One-line description:
# HyperCanny: A novel metaheuristic algorithm for solving black box optimization problems using a combination of grid search, random search, and evolutionary algorithms.

# Code: 