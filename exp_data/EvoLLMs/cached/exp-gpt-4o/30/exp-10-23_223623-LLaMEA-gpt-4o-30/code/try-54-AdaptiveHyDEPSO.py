import numpy as np

class AdaptiveHyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(10 * dim, 50)  # Ensuring a minimum population size
        self.F = 0.5 + 0.2 * np.random.rand()  # Adaptive Differential weight
        self.CR = 0.8  # Stabilized Crossover probability
        self.w = 0.5  # Balanced Inertia weight
        self.c1 = 1.8  # Emphasized cognitive learning
        self.c2 = 1.4  # Increased social learning
        self.velocity_clamp = 0.15 * (self.upper_bound - self.lower_bound)  # Adjusted velocity clamp
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim)) * self.velocity_clamp
        self.personal_best = np.copy(self.population)
        self.personal_best_fitness = np.full(self.population_size, np.inf)
        self.global_best = np.copy(self.population[np.argmin(self.personal_best_fitness)])
        self.best_fitness = np.inf
        self.strategy_dynamic = True  # Enable enhanced dynamic strategy selection

    def __call__(self, func):
        evals = 0
        fitness = np.array([func(ind) for ind in self.population])
        evals += self.population_size

        # Update personal best fitness records
        self.personal_best_fitness = np.minimum(self.personal_best_fitness, fitness)
        idx = self.personal_best_fitness < np.array([func(x) for x in self.personal_best])
        self.personal_best[idx] = self.population[idx]

        # Update global best
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.global_best = self.population[min_idx]
            self.best_fitness = fitness[min_idx]

        while evals < self.budget:
            if self.strategy_dynamic:  # Enhanced dynamic adaptation
                self.F = 0.4 + 0.3 * np.random.rand()  # Adapt within a new range
                self.CR = 0.75 + 0.1 * np.random.rand()  # Adapt within a new range

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
                    if trial_fitness < self.personal_best_fitness[i]:
                        self.personal_best[i] = trial
                        self.personal_best_fitness[i] = trial_fitness

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

        return self.global_best, self.best_fitness