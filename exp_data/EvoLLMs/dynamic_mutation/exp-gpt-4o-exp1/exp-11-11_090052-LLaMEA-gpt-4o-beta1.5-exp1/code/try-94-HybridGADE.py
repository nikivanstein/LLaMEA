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
        self.restart_threshold = 0.2 * self.population_size  # Allow some room for random restarts

    def __call__(self, func):
        # Initialize population fitness
        self.evaluate_population(func)
        
        while self.evaluations < self.budget:
            best_idx = np.argmin(self.fitness)  # Track the best individual
            fitness_variance = np.var(self.fitness)  # Compute fitness variance
            adaptive_mutation_factor = self.mutation_factor * (1 + fitness_variance)  # Adaptive mutation scaling
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                # Differential Evolution mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + adaptive_mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                
                # Crossover
                trial = np.where(np.random.rand(self.dim) < self.crossover_probability, mutant, self.population[i])
                
                # Adaptive neighborhood search
                neighbors_idx = np.random.choice(self.population_size, self.neighborhood_size, replace=False)
                local_best = self.population[neighbors_idx[np.argmin(self.fitness[neighbors_idx])]]
                trial = np.where(np.random.rand(self.dim) < 0.5, trial, local_best + np.random.rand(self.dim) * (trial - local_best))
                
                # Evaluate trial solution
                trial_fitness = func(trial)
                self.evaluations += 1
                
                # Selection using tournament selection
                opponent_idx = np.random.randint(self.population_size)
                if trial_fitness < self.fitness[opponent_idx]:
                    self.population[opponent_idx] = trial
                    self.fitness[opponent_idx] = trial_fitness
            
            # Ensure the best individual is retained
            if np.min(self.fitness) < self.fitness[best_idx]:
                self.population[best_idx], self.fitness[best_idx] = self.population[np.argmin(self.fitness)], np.min(self.fitness)
            
            # Random restart mechanism
            if np.random.rand() < 0.05:  # 5% chance to randomly restart a part of the population
                restart_indices = np.random.choice(self.population_size, int(self.restart_threshold), replace=False)
                self.population[restart_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(restart_indices), self.dim))
                self.evaluate_population(func)  # Re-evaluate those individuals

        # Return the best solution found
        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.evaluations += 1