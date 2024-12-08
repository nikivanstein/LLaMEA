import numpy as np

class DynamicEnsemblePSODEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.5
        self.CR = 0.3
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def mutate(self, x, pop):
        idxs = np.random.choice(len(pop), 3, replace=False)
        a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
        return np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound) if np.random.rand() < self.CR else x

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, size=(self.pop_size, self.dim))
        velocity = np.zeros((self.pop_size, self.dim))
        pbest = population.copy()
        gbest = pbest[np.argmin([func(p) for p in pbest)]
        
        for _ in range(self.budget - self.pop_size):
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocity = self.w * velocity + self.c1 * r1 * (pbest - population) + self.c2 * r2 * (gbest - population)
            population = np.clip(population + velocity, self.lower_bound, self.upper_bound)
            
            for i in range(self.pop_size):
                if np.random.rand() < 0.025:
                    self.F = np.clip(self.F + np.random.normal(0, 0.1), 0, 1)
                    self.CR = np.clip(self.CR + np.random.normal(0, 0.1), 0, 1)
                candidate = self.mutate(population[i], population)
                if func(candidate) < func(population[i]):
                    population[i] = candidate
                    pbest[i] = candidate
                    if func(candidate) < func(gbest):
                        gbest = candidate
        
        return gbest