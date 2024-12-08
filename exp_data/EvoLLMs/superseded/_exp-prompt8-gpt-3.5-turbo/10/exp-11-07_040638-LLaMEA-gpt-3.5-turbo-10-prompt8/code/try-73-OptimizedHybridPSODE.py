import numpy as np

class OptimizedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.w = 0.5 + 0.3 * np.cos(np.linspace(0, 2*np.pi, self.max_iter))
        self.c = 1.5 - np.linspace(0, 1, self.max_iter)
        
    def __call__(self, func):
        def create_population():
            return np.random.uniform(-5.0, 5.0, size=(self.pop_size, self.dim))
        
        def evaluate_population(population):
            return np.array([func(individual) for individual in population])
        
        population = create_population()
        fitness = evaluate_population(population)
        best_idx = np.argmin(fitness)
        global_best = population[best_idx]
        
        for t in range(self.max_iter):
            r1, r2 = np.random.uniform(0, 1, size=(2, self.dim))
            for i in range(self.pop_size):
                vel_term1 = self.w[t] * population[i]
                vel_term2 = self.c[t] * r1 * (global_best - population[i])
                vel_term3 = self.c[t] * r2 * (population[best_idx] - population[i])
                velocity = vel_term1 + vel_term2 + vel_term3
                population[i] += velocity
                population[i] = np.clip(population[i], -5.0, 5.0)
                
                idx = [j for j in range(self.pop_size) if j != i]
                a, b, c = population[np.random.choice(idx, 3, replace=False)]
                mutant = np.clip(a + 0.8 * (b - c), -5.0, 5.0)
                
                fitness_i = func(population[i])
                fitness_m = func(mutant)
                
                if fitness_m < fitness_i:
                    population[i] = mutant
                    fitness_i = fitness_m
                
                if fitness_i < fitness[best_idx]:
                    best_idx = i
                    global_best = population[i]
        
        return global_best