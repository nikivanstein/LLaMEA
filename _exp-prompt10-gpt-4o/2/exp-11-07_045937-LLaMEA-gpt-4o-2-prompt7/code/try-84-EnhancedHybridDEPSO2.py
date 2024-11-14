import numpy as np

class EnhancedHybridDEPSO2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.6  # Adaptive mutation factor for better exploration
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
        idxs = np.random.choice(self.pop_size, (self.pop_size, 3), replace=False)
        a, b, c = population[idxs[:, 0]], population[idxs[:, 1]], population[idxs[:, 2]]
        self.F = 0.5 + 0.3 * np.random.rand(self.pop_size, 1)  # Vectorized adaptive mutation rate
        mutants = np.clip(a + self.F * (b - c), self.bound_min, self.bound_max)
        mask = np.random.rand(self.pop_size, self.dim) < self.CR
        trials = np.where(mask, mutants, population)
        trial_fitness = self.evaluate_population(trials, func)
        improved = trial_fitness < fitness
        population[improved], fitness[improved] = trials[improved], trial_fitness[improved]
        return population, fitness
    
    def particle_swarm_optimization(self, population, fitness, personal_best, personal_best_fitness, global_best, func):
        global_best_fitness = func(global_best)
        inertia_weight = 0.8 - 0.5 * (self.budget - self.pop_size) / self.budget  # Dynamic inertia weight
        r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
        velocity_update = self.c1 * r1 * (personal_best - population) + self.c2 * r2 * (global_best - population)
        self.velocities = inertia_weight * self.velocities + velocity_update
        updated_positions = np.clip(population + self.velocities, self.bound_min, self.bound_max)
        current_fitness = self.evaluate_population(updated_positions, func)
        improved = current_fitness < personal_best_fitness
        personal_best[improved], personal_best_fitness[improved] = updated_positions[improved], current_fitness[improved]
        if current_fitness.min() < global_best_fitness:
            global_best = updated_positions[current_fitness.argmin()]
        return updated_positions, personal_best, personal_best_fitness, global_best
    
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