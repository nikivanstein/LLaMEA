import numpy as np
from scipy.optimize import minimize
import random

class EvolutionaryAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = self.initialize_population()

    def initialize_population(self):
        return [np.random.uniform(-5.0, 5.0, self.dim) for _ in range(self.population_size)]

    def __call__(self, func):
        def objective(x):
            return func(x)
        result = minimize(objective, self.population[0], method="SLSQP", bounds=[(-5.0, 5.0) for _ in range(self.dim)], constraints={"type": "eq", "fun": lambda x: 0})
        if result.success:
            return result.x, result.fun
        else:
            return None, -np.inf

    def mutate(self, individual):
        if random.random() < 0.45:
            return individual + np.random.uniform(-0.1, 0.1, self.dim)
        else:
            return individual

    def select_solution(self, individual):
        return random.choices([individual], weights=self.population, k=1)[0]