import numpy as np
import concurrent.futures

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _evaluate_population(self, func, population):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [executor.submit(func, ind) for ind in population]
        return [result.result() for result in results]

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_scores = self._evaluate_population(func, population)
            elite_indices = np.argsort(fitness_scores)[:2]
            elites = population[elite_indices]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            population[elite_indices[np.argmax(fitness_scores[elite_indices])]] = offspring
        return population[np.argmin(self._evaluate_population(func, population))]