import numpy as np

class HybridGADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.mutation_factor = 0.8
        self.neighborhood_size = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.evaluations = 0

    def __call__(self, func):
        self.evaluate_population(func)
        
        while self.evaluations < self.budget:
            best_idx = np.argmin(self.fitness)
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                if np.random.rand() < 0.5:
                    step_size = self.levy_flight()
                    mutant = a + step_size * (b - c)
                else:
                    mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                
                crossover_rate = 0.9 if self.evaluations < self.budget / 2 else 0.6
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, self.population[i])
                
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

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1

    def levy_flight(self, beta=1.5):
        sigma = (np.gamma(1 + beta) * np.sin(np.pi * beta / 2) /
                 (np.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        return u / np.abs(v) ** (1 / beta)