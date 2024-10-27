import numpy as np

class NovelEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population = np.random.uniform(-5.0, 5.0, (budget, dim))

    def novel_mutation(self, parent, scale=0.1):
        direction = np.random.normal(0, 1, self.dim)
        return parent + direction * scale

    def __call__(self, func):
        for _ in range(self.budget):
            parents = np.random.choice(self.population, size=2, replace=False)
            child = self.novel_mutation(parents[0]) if func(parents[0]) < func(parents[1]) else self.novel_mutation(parents[1])
            if func(child) < func(parents[0]):
                self.population[np.argmax([func(parents[0]), func(child)])] = child
        return self.population[np.argmin([func(ind) for ind in self.population])]