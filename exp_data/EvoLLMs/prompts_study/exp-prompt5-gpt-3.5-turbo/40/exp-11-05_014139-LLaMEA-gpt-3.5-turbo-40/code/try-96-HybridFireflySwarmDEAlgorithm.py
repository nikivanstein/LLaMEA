import numpy as np

class HybridFireflySwarmDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.alpha = 0.5
        self.beta_min = 0.2
        self.beta_max = 0.9
        self.gamma_min = 0.1
        self.gamma_max = 0.9

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        
        for _ in range(self.budget):
            for i in range(self.pop_size):
                idxs = np.random.choice(list(range(self.pop_size)), 3, replace=False)
                x_r1, x_r2, x_r3 = pop[idxs]
                beta = np.random.uniform(self.beta_min, self.beta_max)
                gamma = np.random.uniform(self.gamma_min, self.gamma_max)
                new_pop = np.copy(pop)
                for j in range(self.pop_size):
                    if j != i:
                        attr = np.linalg.norm(pop[i] - pop[j])
                        new_pop[j] += beta * np.exp(-self.alpha * attr**2) * (pop[i] - pop[j]) + gamma * np.random.normal(0, 1, self.dim)
                new_pop_fitness = np.array([func(ind) for ind in new_pop])
                min_idx = np.argmin(new_pop_fitness)
                if new_pop_fitness[min_idx] < fitness[i]:
                    pop[i] = new_pop[min_idx]
                    fitness[i] = new_pop_fitness[min_idx]
        return pop[np.argmin(fitness)]