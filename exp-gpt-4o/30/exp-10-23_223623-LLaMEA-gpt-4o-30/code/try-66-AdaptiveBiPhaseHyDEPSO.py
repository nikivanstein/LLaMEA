import numpy as np

class AdaptiveBiPhaseHyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim  # Slightly smaller population for faster convergence
        self.F = np.random.uniform(0.5, 0.9)  # Adjusted F range for better exploration
        self.CR = np.random.uniform(0.6, 0.9)  # Broader crossover probability
        self.w = 0.6  # Higher inertia weight for improved exploration
        self.c1 = 1.8  # Increased cognitive learning factor
        self.c2 = 1.3  # Decreased social learning factor for balance
        self.velocity_clamp = 0.2 * (self.upper_bound - self.lower_bound)  # Decreased velocity clamp
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim)) * self.velocity_clamp
        self.personal_best = np.copy(self.population)
        self.global_best = np.copy(self.population[np.argmin([float('inf')] * self.population_size)])
        self.best_fitness = float('inf')
        self.strategy_dynamic = True  # Enable dynamic strategy selection
        self.local_search_prob = 0.3  # Probability of local search execution

    def __call__(self, func):
        evals = 0
        fitness = np.array([func(ind) for ind in self.population])
        self.personal_fitness = np.copy(fitness)
        evals += self.population_size

        while evals < self.budget:
            if self.strategy_dynamic:  # Dynamic adaptation
                self.F = 0.5 + 0.3 * np.random.rand()  # Randomize within a range
                self.CR = 0.6 + 0.3 * np.random.rand()  # Randomize within a range

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

            # Local search phase with probability
            if np.random.rand() < self.local_search_prob:
                for i in range(self.population_size):
                    candidate = self.population[i] + np.random.uniform(-0.1, 0.1, self.dim)
                    candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
                    candidate_fitness = func(candidate)
                    evals += 1
                    if candidate_fitness < fitness[i]:
                        self.population[i] = candidate
                        fitness[i] = candidate_fitness
                        if candidate_fitness < self.best_fitness:
                            self.global_best = candidate
                            self.best_fitness = candidate_fitness

        return self.global_best, self.best_fitness