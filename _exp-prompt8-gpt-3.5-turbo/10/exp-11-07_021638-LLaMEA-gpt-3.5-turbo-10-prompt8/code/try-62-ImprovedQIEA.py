import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def evaluate_population(self, func, population):
        return [func(ind) for ind in population]

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                fitness_scores = executor.submit(self.evaluate_population, func, population)
                elites = population[np.argsort(fitness_scores.result())[:2]]
                offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
                population[np.argmax(fitness_scores.result())] = offspring
        return population[np.argmin(self.evaluate_population(func, population))]