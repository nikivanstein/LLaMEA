import numpy as np

class ImprovedQIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 1.0

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            fitness_scores = np.array([func(ind) for ind in population])
            elites = population[np.argsort(fitness_scores)[:2]]
            offspring = np.mean(elites, axis=0) + np.random.normal(0, 1, self.dim) * self.mutation_rate
            worst_idx = np.argmax(fitness_scores)
            population[worst_idx] = offspring
            if _ % (self.budget // 10) == 0:  # Update mutation rate every 10% of budget
                self.mutation_rate *= 0.95  # Reduce mutation rate gradually
        return population[np.argmin([func(ind) for ind in population])]