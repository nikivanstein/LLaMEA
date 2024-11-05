import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(10 * dim, max(20, budget // 50))
        self.initial_mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0
        self.success_mem = []
        self.trial_successes = np.zeros(self.pop_size)
    
    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1
    
    def select_parents(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        return np.random.choice(indices, 3, replace=False)
    
    def mutate(self, idx, r1, r2, r3):
        success_ratio = np.mean(self.success_mem[-5:]) if self.success_mem else 1.0
        mutation_factor = self.initial_mutation_factor * (1 - self.evaluations / self.budget) * success_ratio
        mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3])
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def dynamic_crossover_rate(self, idx):
        return self.crossover_rate + 0.1 * (self.trial_successes[idx] / (np.sum(self.trial_successes) + 1e-9))
    
    def crossover(self, idx, mutant):
        trial = np.copy(self.population[idx])
        cr = self.dynamic_crossover_rate(idx)
        for j in range(self.dim):
            if np.random.rand() <= cr or j == np.random.randint(self.dim):
                trial[j] = mutant[j]
        return trial
    
    def resize_population(self):
        if self.evaluations > self.budget // 2 and self.pop_size > 10:
            self.pop_size = max(10, self.pop_size // 2)
            self.population = self.population[:self.pop_size]
            self.fitness = self.fitness[:self.pop_size]
            self.trial_successes = self.trial_successes[:self.pop_size]
    
    def optimize(self, func):
        self.evaluate_population(func)
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                r1, r2, r3 = self.select_parents(i)
                mutant = self.mutate(i, r1, r2, r3)
                trial = self.crossover(i, mutant)
                
                if self.evaluations < self.budget:
                    trial_fitness = func(trial)
                    self.evaluations += 1
                    
                    if trial_fitness < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness
                        self.success_mem.append(1)
                        self.trial_successes[i] += 1
                    else:
                        self.success_mem.append(0)
                        self.trial_successes[i] = max(0, self.trial_successes[i] - 1)
                    self.success_mem = self.success_mem[-10:]
            
            self.resize_population()
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
    
    def __call__(self, func):
        return self.optimize(func)