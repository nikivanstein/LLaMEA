import numpy as np

class EnhancedAdaptiveHyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Increased population size for diversity
        self.F = 0.5 + 0.3 * np.random.rand()  # Adaptive differential weight
        self.CR = 0.8 + 0.1 * np.random.rand()  # Adaptive crossover probability
        self.w = 0.5  # Adjusted inertia weight for balanced exploration
        self.c1 = 1.5 # Moderated emphasis on cognitive learning
        self.c2 = 1.5 # Balanced social learning
        self.velocity_clamp = 0.3 * (self.upper_bound - self.lower_bound) # Enhanced velocity clamp
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim)) * self.velocity_clamp
        self.personal_best = np.copy(self.population)
        self.global_best = np.copy(self.population[np.argmin([float('inf')] * self.population_size)])
        self.best_fitness = float('inf')
        self.strategy_dynamic = True # Keep dynamic strategy selection
        self.local_search_prob = 0.1 # Probability of applying local search

    def __call__(self, func):
        evals = 0
        fitness = np.array([func(ind) for ind in self.population])
        evals += self.population_size

        while evals < self.budget:
            if self.strategy_dynamic:
                self.F = 0.5 + 0.3 * np.random.rand()
                self.CR = 0.8 + 0.1 * np.random.rand()

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
                    self.personal_best[i] = trial if trial_fitness < func(self.personal_best[i]) else self.personal_best[i]

                # Update global best
                if trial_fitness < self.best_fitness:
                    self.global_best = trial
                    self.best_fitness = trial_fitness

                # Stochastic Local Search
                if np.random.rand() < self.local_search_prob:
                    local_trial = trial + np.random.uniform(-0.1, 0.1, self.dim)
                    local_trial_fitness = func(np.clip(local_trial, self.lower_bound, self.upper_bound))
                    evals += 1
                    if local_trial_fitness < trial_fitness:
                        self.population[i] = local_trial
                        fitness[i] = local_trial_fitness
                        if local_trial_fitness < self.best_fitness:
                            self.global_best = local_trial
                            self.best_fitness = local_trial_fitness

            # Particle Swarm Optimization update
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            cognitive = self.c1 * r1 * (self.personal_best - self.population)
            social = self.c2 * r2 * (self.global_best - self.population)
            self.velocities = self.w * self.velocities + cognitive + social
            self.velocities = np.clip(self.velocities, -self.velocity_clamp, self.velocity_clamp)
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

        return self.global_best, self.best_fitness