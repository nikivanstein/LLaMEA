import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.6  # Adjusted mutation factor
        self.CR = 0.9
        self.c1 = 1.8  # Slightly adjusted PSO parameter
        self.c2 = 2.0
        self.bound_min = -5.0
        self.bound_max = 5.0
        self.velocities = np.zeros((self.pop_size, dim))

    def initialize_population(self):
        return np.random.uniform(self.bound_min, self.bound_max, (self.pop_size, self.dim))

    def evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def differential_evolution(self, population, fitness, func):
        idxs = np.random.choice(self.pop_size, (self.pop_size, 3), replace=True)
        a, b, c = population[idxs[:, 0]], population[idxs[:, 1]], population[idxs[:, 2]]
        mutants = np.clip(a + self.F * (b - c), self.bound_min, self.bound_max)
        trials = np.where(np.random.rand(self.pop_size, self.dim) < self.CR, mutants, population)
        trial_fitness = np.apply_along_axis(func, 1, trials)
        better_indices = trial_fitness < fitness
        population[better_indices], fitness[better_indices] = trials[better_indices], trial_fitness[better_indices]
        return population, fitness

    def particle_swarm_optimization(self, population, fitness, personal_best, personal_best_fitness, global_best, func):
        global_best_fitness = func(global_best)
        r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
        self.velocities = 0.7 * self.velocities + self.c1 * r1 * (personal_best - population) + self.c2 * r2 * (global_best - population)
        updated_positions = np.clip(population + self.velocities, self.bound_min, self.bound_max)
        current_fitnesses = np.apply_along_axis(func, 1, updated_positions)
        improved = current_fitnesses < personal_best_fitness
        personal_best[improved], personal_best_fitness[improved] = updated_positions[improved], current_fitnesses[improved]
        if np.min(current_fitnesses) < global_best_fitness:
            global_best = updated_positions[np.argmin(current_fitnesses)]
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
            population, fitness = self.differential_evolution(population, fitness, func)
            population, personal_best, personal_best_fitness, global_best = self.particle_swarm_optimization(
                population, fitness, personal_best, personal_best_fitness, global_best, func
            )
            num_evaluations += 2 * self.pop_size

        return global_best