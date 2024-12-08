import numpy as np
from scipy.optimize import minimize

class AdaptiveBBOO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, size=(self.population_size, self.dim))
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

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

        # Refine the strategy
        refiner = Refiner(dim)
        refiner.update_population(self.population, self.fitnesses, self.population_history)
        self.population = refiner.get_next_population()

        return self.population

class Refiner:
    def __init__(self, dim):
        self.dim = dim
        self.population = None

    def update_population(self, population, fitnesses, history):
        # Use a simple heuristic to refine the strategy
        new_population = np.zeros((len(population), self.dim))
        for i in range(len(population)):
            # Calculate the new individual as the average of its neighbors
            new_individual = np.mean([population[j] for j in range(len(population)) if i!= j])
            # Add a small random perturbation to the new individual
            new_individual += np.random.normal(0, 0.1, size=self.dim)
            # Clip the new individual to the search space
            new_individual = np.clip(new_individual, -5.0, 5.0)
            new_population[i] = new_individual
        return new_population

# One-line description with the main idea
# Adaptive Black Box Optimization using Evolution Strategies with Refinement