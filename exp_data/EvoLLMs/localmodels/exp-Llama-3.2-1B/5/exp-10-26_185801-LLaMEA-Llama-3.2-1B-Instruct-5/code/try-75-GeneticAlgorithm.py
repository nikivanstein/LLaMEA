import numpy as np

class GeneticAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.search_space = np.linspace(-5.0, 5.0, dim)
        self.population_size = 100
        self.population = np.random.uniform(self.search_space, size=(self.population_size, self.dim))
        self.fitness_scores = np.zeros((self.population_size,))

    def __call__(self, func):
        while self.budget > 0:
            self.budget -= 1
            fitness_scores = self.evaluate_fitness(self.population)
            best_individual = np.argmax(fitness_scores)
            self.population = self.population[fitness_scores!= fitness_scores[best_individual]].copy()
            self.population[best_individual] = func(self.population[best_individual])
            if np.isnan(self.population[best_individual]) or np.isinf(self.population[best_individual]):
                raise ValueError("Invalid function value")
            if self.population[best_individual] < 0 or self.population[best_individual] > 1:
                raise ValueError("Function value must be between 0 and 1")
        return self.population

    def evaluate_fitness(self, individual):
        func_value = individual
        while np.isnan(func_value) or np.isinf(func_value):
            func_value = func_value(self.search_space)
        return func_value

# Description: A novel Genetic Algorithm for Black Box Optimization.
# Code: 