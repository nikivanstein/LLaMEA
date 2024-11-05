import numpy as np

class EnhancedAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(15 * dim, max(25, budget // 40))
        self.initial_mutation_factor = 0.7
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0
        self.success_mem = []
        self.mutation_factor_adjustment = np.ones(self.pop_size) * self.initial_mutation_factor
        self.elite = None
        self.elite_fitness = np.inf

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1
                if self.fitness[i] < self.elite_fitness:
                    self.elite_fitness = self.fitness[i]
                    self.elite = np.copy(self.population[i])

    def select_parents(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        return np.random.choice(indices, 3, replace=False)

    def mutate(self, idx, r1, r2, r3):
        historical_factor = np.random.uniform(0.4, 0.9)
        mutation_factor = np.random.uniform(0.5, 1.5) * self.mutation_factor_adjustment[idx]
        mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3]) * historical_factor
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def dynamic_crossover_rate(self, idx):
        elite_influence = 0.2 * (self.fitness[idx] - self.elite_fitness) / (abs(self.elite_fitness) + 1e-9)
        return self.crossover_rate * (1 - elite_influence)

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
                        self.mutation_factor_adjustment[i] = min(1.5, self.mutation_factor_adjustment[i] * 1.1)
                    else:
                        self.success_mem.append(0)
                        self.mutation_factor_adjustment[i] = max(0.5, self.mutation_factor_adjustment[i] * 0.9)

                    if trial_fitness < self.elite_fitness:
                        self.elite_fitness = trial_fitness
                        self.elite = np.copy(trial)

                    self.success_mem = self.success_mem[-10:]
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]
    
    def __call__(self, func):
        return self.optimize(func)