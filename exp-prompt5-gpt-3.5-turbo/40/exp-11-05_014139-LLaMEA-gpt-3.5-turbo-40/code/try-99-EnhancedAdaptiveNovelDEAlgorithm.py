import numpy as np

class EnhancedAdaptiveNovelDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.cr = 0.5
        self.f = 0.5
        self.sigma = 0.1
        self.sigma_min = 0.01
        self.sigma_max = 0.2
        self.cr_min = 0.2
        self.cr_max = 0.9
        self.crowding_factor = 0.5  # Crowding factor for diversity maintenance

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), 3, replace=False)
                x_r1, x_r2, x_r3 = pop[idxs]
                self.f = np.random.uniform(0.1, 0.9)  # Adaptive F
                self.cr = np.random.uniform(0.2, 0.9)  # Adaptive CR
                self.sigma = np.clip(self.sigma * np.exp(0.1 * np.random.randn()), self.sigma_min, self.sigma_max)  # Adaptive Sigma
                mutant = pop[i] + self.f * (x_r1 - pop[i]) + self.f * (x_r2 - x_r3) + np.random.normal(0, self.sigma, self.dim)
                for j in range(self.dim):
                    if np.random.rand() > self.cr:
                        mutant[j] = pop[i][j]
                mutant_fitness = func(mutant)
                # Crowding-based selection for improved diversity
                crowding_candidates = np.random.choice(list(range(self.pop_size)), int(self.crowding_factor * self.pop_size), replace=False)
                crowding_fitness = np.array([fitness[c] for c in crowding_candidates])
                if mutant_fitness < np.max(crowding_fitness):
                    max_idx = crowding_candidates[np.argmax(crowding_fitness)]
                    pop[max_idx] = mutant
                    fitness[max_idx] = mutant_fitness
        return pop[np.argmin(fitness)]