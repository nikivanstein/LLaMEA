import numpy as np

class DynamicPopulationAdaptationDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size_min = 5
        self.pop_size_max = 20
        self.cr = 0.5
        self.f = 0.5
        self.sigma = 0.1
        self.sigma_min = 0.01
        self.sigma_max = 0.2
        self.cr_min = 0.2
        self.cr_max = 0.9

    def __call__(self, func):
        pop_size = self.pop_size_min
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            pop_size = int(self.pop_size_min + (_ / self.budget) * (self.pop_size_max - self.pop_size_min))
            if len(pop) < pop_size:
                new_inds = np.random.uniform(-5.0, 5.0, (pop_size - len(pop), self.dim))
                pop = np.vstack([pop, new_inds])
                fitness = np.concatenate([fitness, np.array([func(ind) for ind in new_inds])])
            
            for i in range(len(pop)):
                idxs = np.random.choice(list(range(len(pop))), 3, replace=False)
                x_r1, x_r2, x_r3 = pop[idxs]
                self.f = np.random.uniform(0.1, 0.9)  # Adaptive F
                self.cr = np.random.uniform(0.2, 0.9)  # Adaptive CR
                self.sigma = np.clip(self.sigma * np.exp(0.1 * np.random.randn()), self.sigma_min, self.sigma_max)  # Adaptive Sigma
                mutant = pop[i] + self.f * (x_r1 - pop[i]) + self.f * (x_r2 - x_r3) + np.random.normal(0, self.sigma, self.dim)
                for j in range(self.dim):
                    if np.random.rand() > self.cr:
                        mutant[j] = pop[i][j]
                mutant_fitness = func(mutant)
                if mutant_fitness < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = mutant_fitness
        return pop[np.argmin(fitness)]