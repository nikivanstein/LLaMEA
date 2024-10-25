import numpy as np

class DynamicBalancedHyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim
        self.F = np.random.uniform(0.4, 0.9)  # Broadened range for F to enhance exploration
        self.CR = np.random.uniform(0.5, 0.8)  # Broadened CR range for exploration and exploitation balance
        self.w = 0.5  # Slightly increased inertia weight to enhance exploration phase
        self.c1 = 1.5  # Enhanced cognitive learning slightly to maintain diversity
        self.c2 = 1.5  # Balanced social learning to improve stability
        self.velocity_clamp = 0.15 * (self.upper_bound - self.lower_bound)  # Adjusted velocity clamp for finer movements
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim)) * self.velocity_clamp
        self.personal_best = np.copy(self.population)
        self.global_best = np.copy(self.population[np.argmin([float('inf')] * self.population_size)])
        self.best_fitness = float('inf')
        self.strategy_dynamic = True
        self.reinit_threshold = 0.05 * self.population_size # Adjusted reinitialization threshold for finer control

    def __call__(self, func):
        evals = 0
        fitness = np.array([func(ind) for ind in self.population])
        self.personal_fitness = np.copy(fitness)
        evals += self.population_size

        while evals < self.budget:
            if self.strategy_dynamic:
                self.F = 0.4 + 0.5 * np.random.rand()  # Adjusted range for dynamic F adaptation
                self.CR = 0.5 + 0.3 * np.random.rand()  # Adjusted range for dynamic CR adaptation

            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x_r1, x_r2, x_r3 = self.population[indices]
                mutant = np.clip(x_r1 + self.F * (x_r2 - x_r3), self.lower_bound, self.upper_bound)

                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.personal_fitness[i]:
                        self.personal_best[i] = trial
                        self.personal_fitness[i] = trial_fitness

                if trial_fitness < self.best_fitness:
                    self.global_best = trial
                    self.best_fitness = trial_fitness

            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive = self.c1 * r1 * (self.personal_best - self.population)
            social = self.c2 * r2 * (self.global_best - self.population)
            self.velocities = self.w * self.velocities + cognitive + social
            self.velocities = np.clip(self.velocities, -self.velocity_clamp, self.velocity_clamp)
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

            if np.std(fitness) < 1e-5:
                reinit_indices = np.random.choice(self.population_size, int(self.reinit_threshold), replace=False)
                self.population[reinit_indices] = np.random.uniform(self.lower_bound, self.upper_bound, (len(reinit_indices), self.dim))
                fitness[reinit_indices] = [func(ind) for ind in self.population[reinit_indices]]
                evals += len(reinit_indices)

        return self.global_best, self.best_fitness