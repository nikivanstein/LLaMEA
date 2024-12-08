import numpy as np

class EnhancedAdaptiveHyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim  # Population size carefully chosen for balance
        self.F = np.random.uniform(0.4, 0.9)  # Wider range for Differential weight
        self.CR = np.random.uniform(0.5, 0.9) # Wider range for crossover probability
        self.w = 0.5  # Moderate inertia weight for balanced exploration and exploitation
        self.c1 = 1.7 # Enhanced cognitive component for personal exploration
        self.c2 = 1.3 # Reduced social learning for diversity
        self.velocity_clamp = 0.3 * (self.upper_bound - self.lower_bound) # Increased velocity clamp
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim)) * self.velocity_clamp
        self.personal_best = np.copy(self.population)
        self.global_best = np.copy(self.population[np.argmin([float('inf')] * self.population_size)])
        self.best_fitness = float('inf')
        self.strategy_dynamic = True  # Maintain dynamic strategy adaptation
        self.reinit_threshold = 0.15 * self.population_size  # Increased threshold for reinitialization

    def __call__(self, func):
        evals = 0
        fitness = np.array([func(ind) for ind in self.population])
        self.personal_fitness = np.copy(fitness)
        evals += self.population_size

        while evals < self.budget:
            if self.strategy_dynamic:  # Enhanced dynamic adaptation
                self.F = 0.4 + 0.5 * np.random.rand()  # Wider randomization range
                self.CR = 0.5 + 0.4 * np.random.rand()  # Wider randomization range

            for i in range(self.population_size):
                # Differential Evolution mutation
                indices = np.random.choice(self.population_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                mutant = np.clip(x_r1 + self.F * (x_r2 - x_r3), self.lower_bound, self.upper_bound)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.personal_fitness[i]:
                        self.personal_best[i] = trial
                        self.personal_fitness[i] = trial_fitness

                # Update global best
                if trial_fitness < self.best_fitness:
                    self.global_best = trial
                    self.best_fitness = trial_fitness

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive = self.c1 * r1 * (self.personal_best - self.population)
            social = self.c2 * r2 * (self.global_best - self.population)
            self.velocities = self.w * self.velocities + cognitive + social
            self.velocities = np.clip(self.velocities, -self.velocity_clamp, self.velocity_clamp)
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

            # Neighborhood-based Reinitialization
            if np.std(fitness) < 1e-5:
                reinit_indices = np.random.choice(self.population_size, int(self.reinit_threshold), replace=False)
                for idx in reinit_indices:
                    neighbors = np.random.choice(self.population_size, 3, replace=False)
                    self.population[idx] = np.mean(self.population[neighbors], axis=0) + np.random.normal(0, 0.1, self.dim)
                    self.population[idx] = np.clip(self.population[idx], self.lower_bound, self.upper_bound)
                    fitness[idx] = func(self.population[idx])
                evals += len(reinit_indices)

        return self.global_best, self.best_fitness