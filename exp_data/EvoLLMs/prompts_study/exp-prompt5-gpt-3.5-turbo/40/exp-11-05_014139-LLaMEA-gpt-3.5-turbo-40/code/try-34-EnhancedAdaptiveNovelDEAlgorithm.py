import numpy as np

class EnhancedAdaptiveNovelDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.cr = 0.5
        self.f = 0.5
        self.sigma = 0.1
        self.sigma_min = 0.01
        self.sigma_max = 0.2
        self.cr_min = 0.2
        self.cr_max = 0.9

    def __call__(self, func):
        self.pop_size = 10  # Dynamic population size
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
                if mutant_fitness < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = mutant_fitness
                    
            # Adapt population size dynamically based on performance
            best_fitness_idx = np.argmin(fitness)
            if np.random.rand() < 0.1:  # Adjust population size with a 10% probability
                self.pop_size = min(50, int(self.pop_size * (1 + np.random.normal(0, 0.1))))
                pop = np.vstack((pop, np.random.uniform(-5.0, 5.0, (self.pop_size - len(pop), self.dim))))
                fitness = np.concatenate((fitness, [func(pop[i]) for i in range(len(pop) - self.pop_size)]))
        return pop[best_fitness_idx]