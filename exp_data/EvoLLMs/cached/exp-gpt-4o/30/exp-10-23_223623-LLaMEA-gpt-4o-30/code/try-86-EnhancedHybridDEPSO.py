import numpy as np

class EnhancedHybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim  # Slightly increased population size for better diversity
        self.F = np.random.uniform(0.4, 0.9)  # Expanded range for differential weight
        self.CR = np.random.uniform(0.5, 0.9)  # Broadened crossover probability range
        self.w = 0.5  # Adjusted inertia weight for improved convergence
        self.c1 = 1.5  # Balanced cognitive learning factor
        self.c2 = 1.7  # Increased social learning factor for better global influence
        self.velocity_clamp = 0.3 * (self.upper_bound - self.lower_bound)  # Adjusted velocity clamp
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim)) * self.velocity_clamp
        self.personal_best = np.copy(self.population)
        self.global_best = np.copy(self.population[np.argmin([float('inf')] * self.population_size)])
        self.best_fitness = float('inf')
        self.strategy_dynamic = np.random.rand() < 0.3  # Probabilistic dynamic strategy
        self.reinit_threshold = 0.2 * self.population_size  # Higher threshold for reinitialization

    def __call__(self, func):
        evals = 0
        fitness = np.array([func(ind) for ind in self.population])
        self.personal_fitness = np.copy(fitness)
        evals += self.population_size

        while evals < self.budget:
            if self.strategy_dynamic:  # Dynamic adaptation
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

            # Reinitialization of a portion of the population if stagnation is detected
            if np.std(fitness) < 1e-5:
                reinit_indices = np.random.choice(self.population_size, int(self.reinit_threshold), replace=False)
                self.population[reinit_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(reinit_indices), self.dim))
                fitness[reinit_indices] = [func(ind) for ind in self.population[reinit_indices]]
                evals += len(reinit_indices)

        return self.global_best, self.best_fitness