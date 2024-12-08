import numpy as np

class HybridDEPSOAlgorithm:
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
        self.w = 0.5
        self.c1 = 1.49445
        self.c2 = 1.49445

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
                mutant_de = pop[i] + self.f * (x_r1 - pop[i]) + self.f * (x_r2 - x_r3) + np.random.normal(0, self.sigma, self.dim)
                
                # Particle Swarm Optimization
                v = self.w * v + self.c1 * np.random.rand() * (pbest - pop[i]) + self.c2 * np.random.rand() * (gbest - pop[i])
                pop_pso = pop[i] + v
                
                mutant = np.where(np.random.rand(self.dim) > self.cr, mutant_de, pop_pso)
                
                mutant_fitness = func(mutant)
                if mutant_fitness < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = mutant_fitness
        return pop[np.argmin(fitness)]