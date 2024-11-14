import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.w = 0.5 + 0.3 * np.cos(np.linspace(0, 2*np.pi, self.max_iter))
        self.c = 1.5 - np.linspace(0, 1, self.max_iter)
        self.population = np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))

    def __call__(self, func):
        fitness = np.array([func(individual) for individual in self.population])
        best_idx = np.argmin(fitness)
        global_best = self.population[best_idx]
        
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                r1, r2 = np.random.uniform(0, 1, size=(2, self.dim))
                velocity = self.w[t] * self.population[i] + self.c[t] * r1 * (global_best - self.population[i]) + self.c[t] * r2 * (self.population[best_idx] - self.population[i])
                self.population[i] += velocity
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                
                idx = [j for j in range(self.pop_size) if j != i]
                a, b, c = self.population[np.random.choice(idx, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), -5.0, 5.0)
                
                fitness_i = func(self.population[i])
                fitness_m = func(mutant)
                
                if fitness_m < fitness_i:
                    self.population[i] = mutant
                    fitness_i = fitness_m
                
                if fitness_i < fitness[best_idx]:
                    best_idx = i
                    global_best = self.population[i]
        
        return global_best