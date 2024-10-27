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
        self.iterations = 0

    def __call__(self, func):
        def eval_func(x):
            return func(x)

        def evaluate_budget(func, x, budget):
            if budget <= 0:
                raise ValueError("Budget cannot be zero or negative")
            return np.sum([eval_func(x + np.random.normal(0, 1, size=self.dim)) for _ in range(budget)])

        def fitness(individual):
            fitness = evaluate_budget(eval_func, individual, self.budget)
            self.fitnesses[individual] = fitness
            self.population_history.append(individual)
            return fitness

        def optimize(individual):
            individual = np.array(individual)
            result = minimize(lambda x: -fitness(individual), individual, method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim)
            return result.x

        def update_individual(individual):
            individual = np.array(individual)
            result = optimize(individual)
            updated_individual = individual + np.random.normal(0, 1, size=self.dim)
            return updated_individual

        def update_population():
            new_population = np.zeros((self.population_size, self.dim))
            for i in range(self.population_size):
                new_individual = update_individual(self.population[i])
                new_population[i] = new_individual
            return new_population

        def select_population(population):
            return np.argsort(population, axis=0)[:self.population_size // 2]

        def select_fittest(population):
            return select_population(population)

        self.population = select_fittest(population)
        self.fitnesses = np.zeros((self.population_size, self.dim))
        self.population_history = []

        return self.population