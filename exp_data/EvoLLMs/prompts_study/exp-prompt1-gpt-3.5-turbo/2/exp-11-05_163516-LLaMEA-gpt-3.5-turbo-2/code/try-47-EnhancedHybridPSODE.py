import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 20
        self.max_iter = budget // self.pop_size
        self.w = 0.5
        self.c1 = 1.5
        self.c2 = 1.5
        self.cr = 0.9
        self.f = 0.8
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
    
    def __call__(self, func):
        best_position = self.population[np.argmin([func(ind) for ind in self.population])]
        best_fitness = func(best_position)
        chaos_factor = 0.1
        
        for _ in range(self.max_iter):
            for i in range(self.pop_size):
                # Introducing chaotic search for exploration
                chaos = np.random.uniform(-chaos_factor, chaos_factor, self.dim)
                self.population[i] += chaos
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                
                # PSO update
                r1, r2 = np.random.uniform(0, 1, 2)
                new_velocity = self.w * self.population[i] + self.c1 * r1 * (best_position - self.population[i]) + self.c2 * r2 * (best_position - self.population[i])
                new_position = self.population[i] + new_velocity
                new_position = np.clip(new_position, -5.0, 5.0)
                
                # DE update
                rand_indexes = np.random.choice(np.arange(self.pop_size), 3, replace=False)
                mutant = self.population[rand_indexes[0]] + self.f * (self.population[rand_indexes[1]] - self.population[rand_indexes[2]])
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, self.population[i])
                
                if func(trial) < func(self.population[i]):
                    self.population[i] = trial
                if func(new_position) < func(self.population[i]):
                    self.population[i] = new_position
                
                # Update the global best
                if func(self.population[i]) < best_fitness:
                    best_position = self.population[i]
                    best_fitness = func(best_position)
        
        return best_position