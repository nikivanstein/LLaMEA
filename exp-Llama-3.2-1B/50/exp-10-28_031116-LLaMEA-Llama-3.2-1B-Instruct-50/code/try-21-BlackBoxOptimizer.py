import numpy as np
from scipy.optimize import minimize

class BlackBoxOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 100
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.fitness_scores = np.zeros(self.population_size)
        self.best_individual = None

    def __call__(self, func):
        for _ in range(self.budget):
            func = np.vectorize(func)(self.population)
            result = minimize(func, 0, method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim)
            self.fitness_scores[_] = result.fun
            if result.success:
                self.best_individual = self.population[_]
                break
        return self.best_individual

    def select_solution(self):
        if self.best_individual is None:
            return None
        return np.random.choice(self.population, p=self.fitness_scores / self.fitness_scores.sum())

    def update(self):
        solution = self.select_solution()
        func = np.vectorize(self.func)(solution)
        result = minimize(func, 0, method="SLSQP", bounds=[(-5.0, 5.0)] * self.dim)
        self.fitness_scores[result.x] = result.fun
        if result.success:
            self.best_individual = solution
        return solution