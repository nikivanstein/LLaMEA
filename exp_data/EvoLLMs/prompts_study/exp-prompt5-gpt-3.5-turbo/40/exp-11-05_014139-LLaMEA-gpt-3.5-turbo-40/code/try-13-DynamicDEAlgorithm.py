import numpy as np

class DynamicDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.sigma = 0.1

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            f = np.clip(0.5 - 0.4 * _ / self.budget, 0, 0.5)  # Dynamic adjustment of mutation factor
            cr = np.clip(0.5 - 0.4 * _ / self.budget, 0, 0.5)  # Dynamic adjustment of crossover rate
            for i in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), 3, replace=False)
                x_r1, x_r2, x_r3 = pop[idxs]
                mutant = pop[i] + f * (x_r1 - pop[i]) + f * (x_r2 - x_r3) + np.random.normal(0, self.sigma, self.dim)
                for j in range(self.dim):
                    if np.random.rand() > cr:
                        mutant[j] = pop[i][j]
                mutant_fitness = func(mutant)
                if mutant_fitness < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = mutant_fitness
        return pop[np.argmin(fitness)]