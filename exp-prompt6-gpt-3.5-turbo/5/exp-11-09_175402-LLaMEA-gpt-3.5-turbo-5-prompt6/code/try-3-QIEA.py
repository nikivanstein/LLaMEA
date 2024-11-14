import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            offspring = np.zeros((self.budget, self.dim))
            for i in range(self.budget):
                idx = np.random.randint(0, self.budget, 2)
                parent1, parent2 = population[idx]
                mask = np.random.choice([0, 1], size=self.dim)
                offspring[i] = parent1 * mask + parent2 * (1 - mask)
            population = np.where(np.array([func(ind) for ind in offspring]) < np.array([func(ind) for ind in population]), offspring, population)
        return population[np.argmin([func(ind) for ind in population])]