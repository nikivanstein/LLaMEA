import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.neighborhood_size = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0
        self.restart_threshold = 0.2 * self.population_size

    def __call__(self, func):
        self.evaluate_population(func)
        
        while self.evaluations < self.budget:
            best_idx = np.argmin(self.fitness)
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                adaptive_mut_factor = np.random.uniform(0.5, 1.0)  # Adaptive mutation factor
                mutant = np.clip(a + adaptive_mut_factor * (b - c), self.lower_bound, self.upper_bound)
                
                trial = np.where(np.random.rand(self.dim) < self.crossover_probability, mutant, self.population[i])
                
                neighbors_idx = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
                local_best = self.population[neighbors_idx[np.argmin(self.fitness[neighbors_idx])]]
                trial = np.where(np.random.rand(self.dim) < 0.5, trial, local_best + np.random.rand(self.dim) * (trial - local_best))
                
                trial_fitness = func(trial)
                self.evaluations += 1
                
                opponent_idx = np.random.randint(self.population_size)
                if trial_fitness < self.fitness[opponent_idx]:
                    self.population[opponent_idx] = trial
                    self.fitness[opponent_idx] = trial_fitness
            
            if np.min(self.fitness) < self.fitness[best_idx]:
                self.population[best_idx], self.fitness[best_idx] = self.population[np.argmin(self.fitness)], np.min(self.fitness)
            
            if np.random.rand() < 0.05:
                restart_indices = np.random.choice(self.population_size, int(self.restart_threshold), replace=False)
                self.population[restart_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(restart_indices), self.dim))
                self.evaluate_population(func)
            
            elite_preservation_idx = np.argmin(self.fitness)
            self.population[elite_preservation_idx] = self.population[np.argmin(self.fitness)]
            self.fitness[elite_preservation_idx] = np.min(self.fitness)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1