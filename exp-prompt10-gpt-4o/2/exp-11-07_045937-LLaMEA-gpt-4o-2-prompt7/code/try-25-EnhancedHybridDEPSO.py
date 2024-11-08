import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.bound_min = -5.0
        self.bound_max = 5.0
        self.velocities = np.zeros((self.pop_size, dim))
    
    def initialize_population(self):
        return np.random.uniform(self.bound_min, self.bound_max, (self.pop_size, self.dim))
    
    def evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])
    
    def differential_evolution(self, population, fitness, func):
        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            a, b, c = population[idxs]
            mutant = a + self.F * (b - c)
            trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
            trial = np.clip(trial, self.bound_min, self.bound_max)  # Mutation outside loop
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i], fitness[i] = trial, trial_fitness
        return population, fitness
    
    def particle_swarm_optimization(self, population, fitness, personal_best, personal_best_fitness, global_best, func):
        global_best_fitness = func(global_best)
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            self.velocities[i] = 0.7 * self.velocities[i] + self.c1 * r1 * (personal_best[i] - population[i]) + self.c2 * r2 * (global_best - population[i])
            population[i] = np.clip(population[i] + self.velocities[i], self.bound_min, self.bound_max)
            current_fitness = func(population[i])
            if current_fitness < personal_best_fitness[i]:
                personal_best[i], personal_best_fitness[i] = population[i], current_fitness
                if current_fitness < global_best_fitness:
                    global_best, global_best_fitness = population[i], current_fitness
        return population, personal_best, personal_best_fitness, global_best
    
    def __call__(self, func):
        np.random.seed()
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)
        num_evaluations = self.pop_size
        
        personal_best = np.copy(population)
        personal_best_fitness = np.copy(fitness)
        global_best = population[np.argmin(fitness)]
        
        while num_evaluations < self.budget:
            population, fitness = self.differential_evolution(population, fitness, func)
            population, personal_best, personal_best_fitness, global_best = self.particle_swarm_optimization(
                population, fitness, personal_best, personal_best_fitness, global_best, func
            )
            num_evaluations += 2 * self.pop_size
        
        return global_best