import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            offspring = np.zeros((self.budget, self.dim))
            fitness_values = np.array([func(ind) for ind in population])
            max_fitness = max(fitness_values)
            mutation_rate = 0.1 + 0.4 * (max_fitness - fitness_values) / max_fitness
            for i in range(self.budget):
                idx = np.random.randint(0, self.budget, 2)
                parent1, parent2 = population[idx]
                mask = np.random.choice([0, 1], size=self.dim, p=[mutation_rate[i], 1 - mutation_rate[i]])
                offspring[i] = parent1 * mask + parent2 * (1 - mask)
            population = np.where(np.array([func(ind) for ind in offspring]) < np.array([func(ind) for ind in population]), offspring, population)
        return population[np.argmin([func(ind) for ind in population])]