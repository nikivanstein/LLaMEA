import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        mutation_rate = 0.5  # Initial mutation rate
        for _ in range(self.budget):
            offspring = np.zeros((self.budget, self.dim))
            for i in range(self.budget):
                idx = np.random.randint(0, self.budget, 2)
                parent1, parent2 = population[idx]
                mask = np.random.choice([0, 1], size=self.dim, p=[mutation_rate, 1 - mutation_rate])
                offspring[i] = parent1 * mask + parent2 * (1 - mask)
            population = np.where(np.array([func(ind) for ind in offspring]) < np.array([func(ind) for ind in population]), offspring, population)
            mutation_rate = max(0.1, mutation_rate * 0.95)  # Update mutation rate dynamically
        return population[np.argmin([func(ind) for ind in population])]