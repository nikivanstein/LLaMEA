import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        evals = [func(ind) for ind in population]
        for _ in range(self.budget):
            elites = population[np.argsort(evals)[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim)
            eval_offspring = func(offspring)
            max_idx = np.argmax(evals)
            if eval_offspring < evals[max_idx]:
                population[max_idx] = offspring
                evals[max_idx] = eval_offspring
        return population[np.argmin(evals)]