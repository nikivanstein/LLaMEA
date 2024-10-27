import numpy as np
from scipy.optimize import minimize

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

    def mutate(self, individual):
        new_individual = individual.copy()
        if np.random.rand() < 0.05:
            new_individual[self.search_space] = np.random.uniform(self.search_space.min(), self.search_space.max())
        return new_individual

    def evaluate_fitness(self, individual):
        func_value = self.__call__(individual)
        if np.isnan(func_value) or np.isinf(func_value):
            raise ValueError("Invalid function value")
        if func_value < 0 or func_value > 1:
            raise ValueError("Function value must be between 0 and 1")
        return func_value

    def __next_generation(self, population):
        next_generation = population.copy()
        for _ in range(self.dim):
            new_individual = next_generation[np.random.randint(len(next_generation))]
            next_generation.append(self.mutate(new_individual))
        return next_generation

    def run(self, population_size, generations):
        for _ in range(generations):
            population = self.__next_generation(population_size)
            fitness_values = [self.evaluate_fitness(individual) for individual in population]
            best_individual = population[np.argmin(fitness_values)]
            self.func_evaluations += fitness_values[np.argmin(fitness_values)]
            print(f"Generation {_+1}, Best Individual: {best_individual}, Best Fitness: {self.evaluate_fitness(best_individual)}")