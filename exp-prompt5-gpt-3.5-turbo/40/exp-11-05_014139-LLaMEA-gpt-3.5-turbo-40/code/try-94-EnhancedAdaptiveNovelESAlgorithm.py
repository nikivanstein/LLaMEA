import numpy as np

class EnhancedAdaptiveNovelESAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.sigma = 0.1
        self.sigma_min = 0.01
        self.sigma_max = 0.2
        self.mu = np.random.uniform(-5.0, 5.0, dim)
        self.sigma_matrix = np.eye(dim)
        self.c_sigma = 0.1
        self.d_sigma = 1 + dim**0.5
        self.chi = dim**0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
    def __call__(self, func):
        pop = np.random.multivariate_normal(self.mu, self.sigma_matrix, self.pop_size)
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            offspring = np.random.multivariate_normal(self.mu, self.sigma_matrix, self.pop_size)
            offspring_fitness = np.array([func(ind) for ind in offspring])
            idxs = np.where(offspring_fitness < fitness)
            pop[idxs] = offspring[idxs]
            fitness[idxs] = offspring_fitness[idxs]
            
            fitness_order = np.argsort(fitness)
            elite = pop[fitness_order[:self.pop_size // 2]]
            
            self.mu = np.mean(elite, axis=0)
            diff = elite - self.mu
            self.sigma_matrix = (1 - self.c_sigma) * self.sigma_matrix + (self.c_sigma / self.d_sigma) * (diff.T @ diff) + self.chi * np.random.normal(0, 1, (self.dim, self.dim))            
        return self.mu