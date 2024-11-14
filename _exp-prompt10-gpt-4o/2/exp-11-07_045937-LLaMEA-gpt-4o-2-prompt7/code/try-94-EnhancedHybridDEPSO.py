import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.6
        self.CR = 0.9
        self.c1 = 1.8
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
            self.F = 0.5 + 0.3 * np.random.rand()  # Adaptive mutation rate
            mutant = np.clip(a + self.F * (b - c), self.bound_min, self.bound_max)
            trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                population[i], fitness[i] = trial, trial_fitness
        return population, fitness
    
    def particle_swarm_optimization(self, population, fitness, personal_best, personal_best_fitness, global_best, func):
        global_best_fitness = func(global_best)
        inertia_weight = 0.9 - 0.4 * (self.budget - self.pop_size) / self.budget  # Refined inertia weight
        for i in range(self.pop_size):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            velocity_update = self.c1 * r1 * (personal_best[i] - population[i]) + self.c2 * r2 * (global_best - population[i])
            self.velocities[i] = inertia_weight * self.velocities[i] + velocity_update
            updated_position = np.clip(population[i] + self.velocities[i], self.bound_min, self.bound_max)
            current_fitness = func(updated_position)
            if current_fitness < personal_best_fitness[i]:
                personal_best[i], personal_best_fitness[i] = updated_position, current_fitness
                if current_fitness < global_best_fitness:
                    global_best, global_best_fitness = updated_position, current_fitness
            population[i] = updated_position
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
            if num_evaluations % (2 * self.pop_size) < self.pop_size: 
                population, fitness = self.differential_evolution(population, fitness, func)
            else:
                population, personal_best, personal_best_fitness, global_best = self.particle_swarm_optimization(
                    population, fitness, personal_best, personal_best_fitness, global_best, func
                )
            num_evaluations += self.pop_size
        
        return global_best