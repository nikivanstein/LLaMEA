import numpy as np

class QIEA:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.local_search_rate = 0.2

    def local_search(self, population, func):
        num_local_search = int(self.budget * self.local_search_rate)
        best_idx = np.argmin([func(ind) for ind in population])
        best_solution = population[best_idx]
        for _ in range(num_local_search):
            new_solution = best_solution + np.random.normal(0, 0.1, self.dim)
            if func(new_solution) < func(best_solution):
                best_solution = new_solution
        return best_solution

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
            population[np.argmin([func(ind) for ind in population])] = self.local_search(population, func)
        return population[np.argmin([func(ind) for ind in population])]