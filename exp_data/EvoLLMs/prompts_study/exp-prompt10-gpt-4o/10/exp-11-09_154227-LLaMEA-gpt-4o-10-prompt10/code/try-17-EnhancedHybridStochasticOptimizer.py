import numpy as np

class EnhancedHybridStochasticOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.scale_factor = 0.8
        self.crossover_prob = 0.7
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
    
    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.evaluations += 1
    
    def select_parents(self, idx):
        indices = [i for i in range(self.population_size) if i != idx]
        return np.random.choice(indices, 3, replace=False)
    
    def differential_evolution_mutation(self, idx):
        a, b, c = self.select_parents(idx)
        F_dynamic = self.scale_factor + np.random.rand() * 0.2  # Adaptive scaling
        mutant_vector = self.population[a] + F_dynamic * (self.population[b] - self.population[c])
        return np.clip(mutant_vector, self.lower_bound, self.upper_bound)
    
    def crossover(self, target_vector, mutant_vector):
        crossover_mask = np.random.rand(self.dim) < self.crossover_prob
        return np.where(crossover_mask, mutant_vector, target_vector)
    
    def local_search(self, vector):
        perturbation = np.random.normal(0, 0.1, self.dim)
        candidate_vector = vector + perturbation
        return np.clip(candidate_vector, self.lower_bound, self.upper_bound)
    
    def competitive_selection(self, trial_vector, trial_fitness, idx):
        if trial_fitness < self.fitness[idx]:
            self.population[idx] = trial_vector
            self.fitness[idx] = trial_fitness
        elif np.random.rand() < 0.1:  # Explore less fit solutions with low probability
            self.population[idx] = trial_vector
            self.fitness[idx] = trial_fitness
    
    def __call__(self, func):
        self.evaluate_population(func)
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                
                mutant_vector = self.differential_evolution_mutation(i)
                trial_vector = self.crossover(self.population[i], mutant_vector)
                trial_fitness = func(trial_vector)
                self.evaluations += 1
                
                self.competitive_selection(trial_vector, trial_fitness, i)
                
                if np.random.rand() < 0.2:
                    candidate_vector = self.local_search(self.population[i])
                    candidate_fitness = func(candidate_vector)
                    self.evaluations += 1
                    self.competitive_selection(candidate_vector, candidate_fitness, i)
        
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx]