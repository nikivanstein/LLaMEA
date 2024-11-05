import numpy as np

class HybridDifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.pop_size = min(10 * dim, max(20, budget // 50))  # Adaptive pop size
        self.initial_mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.evaluations = 0
        self.elite_individual = None
        self.elite_fitness = np.inf
        
    def evaluate_population(self, func):
        for i in range(self.pop_size):
            if self.evaluations < self.budget:
                self.fitness[i] = func(self.population[i])
                if self.fitness[i] < self.elite_fitness:
                    self.elite_fitness = self.fitness[i]
                    self.elite_individual = np.copy(self.population[i])
                self.evaluations += 1

    def select_parents(self, idx):
        indices = list(range(self.pop_size))
        indices.remove(idx)
        return np.random.choice(indices, 3, replace=False)
    
    def mutate(self, idx, r1, r2, r3):
        mutation_factor = 0.5 + (self.initial_mutation_factor - 0.5) * (1 - self.evaluations / self.budget)  # Adjusted mutation factor
        mutant = self.population[r1] + mutation_factor * (self.population[r2] - self.population[r3])
        return np.clip(mutant, self.lower_bound, self.upper_bound)
    
    def crossover(self, idx, mutant):
        trial = np.copy(self.population[idx])
        j_rand = np.random.randint(self.dim)
        for j in range(self.dim):
            if np.random.rand() <= self.crossover_rate or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def optimize(self, func):
        self.evaluate_population(func)
        
        while self.evaluations < self.budget:
            for i in range(self.pop_size):
                if np.random.rand() > 0.75:  # Local search probability
                    trial = self.local_search(self.population[i])
                else:
                    r1, r2, r3 = self.select_parents(i)
                    mutant = self.mutate(i, r1, r2, r3)
                    trial = self.crossover(i, mutant)

                if self.evaluations < self.budget:
                    trial_fitness = func(trial)
                    self.evaluations += 1
                    
                    if trial_fitness < self.fitness[i]:
                        self.population[i] = trial
                        self.fitness[i] = trial_fitness
                    if trial_fitness < self.elite_fitness:
                        self.elite_individual = trial
                        self.elite_fitness = trial_fitness
        
        return self.elite_individual
    
    def local_search(self, individual):
        # Simple local perturbation
        return np.clip(individual + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)
    
    def __call__(self, func):
        return self.optimize(func)