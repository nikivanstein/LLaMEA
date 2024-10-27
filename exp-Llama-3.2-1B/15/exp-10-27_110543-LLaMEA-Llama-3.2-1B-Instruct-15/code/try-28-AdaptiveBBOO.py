import numpy as np
from scipy.optimize import differential_evolution

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []
        self.algorithms = []

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        for _ in range(self.budget):
            for i in range(self.population_size):
                fitness = evaluate_budget(eval_func, self.population[i], self.budget)
                self.fitnesses[i] = fitness
                self.population_history.append(self.population[i])

        # Select the fittest individuals
        self.population = self.population[np.argsort(self.fitnesses, axis=0)]
        self.fitnesses = self.fitnesses[np.argsort(self.fitnesses, axis=0)]

        # Evolve the population
        for _ in range(100):
            next_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                next_population[i] = self.population[i] + np.random.normal(0, 1, size=self.dim)
                fitness = evaluate_budget(eval_func, next_population[i], self.budget)
                next_population[i] = next_population[i][np.argsort(self.fitnesses, axis=0)]
                self.population[i] = next_population[i]

        # Select the best individual
        best_individual = np.argmax(self.fitnesses)
        best_individual = self.population[best_individual]

        # Refine the strategy
        if np.random.rand() < 0.15:
            new_individual = best_individual
            new_individual = self.evaluate_fitness(new_individual)
            new_individual = np.random.uniform(-5.0, 5.0, size=self.dim)
            new_individual = self.evaluate_fitness(new_individual)
            new_individual = best_individual
            new_individual = np.random.uniform(-5.0, 5.0, size=self.dim)
            new_individual = self.evaluate_fitness(new_individual)
            self.algorithms.append((best_individual, new_individual, eval_func, evaluate_budget, self.budget, self.dim))

        return best_individual

    def evaluate_fitness(self, individual):
        return self.f(individual, self.logger)