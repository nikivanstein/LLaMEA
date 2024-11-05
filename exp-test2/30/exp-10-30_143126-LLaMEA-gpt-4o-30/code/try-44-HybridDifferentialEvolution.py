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
        np.random.shuffle(indices)
        return indices[:3]
    
    def mutate(self, idx, r1, r2, r3):
        dynamic_factor = 0.5 + 0.3 * np.mean(self.success_mem[-5:]) if self.success_mem else 1.0
        mutation_factor = self.initial_mutation_factor * (1 - self.evaluations / self.budget) * dynamic_factor
        mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3])
        shrink_factor = np.random.rand(self.dim) * 0.5  # Introduced shrink factor for fine-grained exploration
        mutant = mutant * shrink_factor + self.population[idx] * (1 - shrink_factor)
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def dynamic_crossover_rate(self, idx):
        return self.crossover_rate + 0.1 * (self.trial_successes[idx] / (np.sum(self.trial_successes) + 1e-9))
    
    def crossover(self, idx, mutant):
        trial = np.copy(self.population[idx])
        j_rand = np.random.randint(self.dim)
        cr = self.dynamic_crossover_rate(idx)
        for j in range(self.dim):
            if np.random.rand() <= cr or j == j_rand:
                trial[j] = mutant[j]
        return trial
    
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
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
    
    def __call__(self, func):
        return self.optimize(func)