import numpy as np

class HEBBO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            func_evaluations += 1
            func_value = func(self.search_space)
            if np.isnan(func_value) or np.isinf(func_value):
                raise ValueError("Invalid function value")
            if func_value < 0 or func_value > 1:
                raise ValueError("Function value must be between 0 and 1")
            self.search_space = np.linspace(-5.0, 5.0, self.dim)
        return func_value

class HEBBOMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.func_evaluations = 0

    def __call__(self, func, initial_individual, population_size, mutation_rate, learning_rate):
        population = initial_individual
        for _ in range(100):
            for _ in range(population_size):
                new_individual = population[np.random.randint(0, population_size)]
                fitness_value = func(new_individual)
                if np.isnan(fitness_value) or np.isinf(fitness_value):
                    raise ValueError("Invalid function value")
                if fitness_value < 0 or fitness_value > 1:
                    raise ValueError("Function value must be between 0 and 1")
                updated_individual = new_individual
                if np.random.rand() < mutation_rate:
                    updated_individual = update_individual(updated_individual, func, learning_rate)
                population.append(updated_individual)
            population = self.filter_population(population, func, self.budget, self.search_space)
        return population

def update_individual(individual, func, learning_rate):
    new_individual = individual
    for _ in range(10):
        new_individual = func(new_individual, learning_rate)
        if np.isnan(new_individual) or np.isinf(new_individual):
            raise ValueError("Invalid function value")
        if new_individual < 0 or new_individual > 1:
            raise ValueError("Function value must be between 0 and 1")
    return new_individual

def filter_population(population, func, budget, search_space):
    return np.array([individual for individual in population if func(individual) < budget])

# Example usage:
HEBBOMetaheuristic(100, 10).__call__(HEBBO(100, 10), np.array([1.0, 2.0, 3.0]), 100, 0.1, 0.5)