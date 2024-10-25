import numpy as np

class SynergisticDEPSOHybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 8 * dim  # Reduced size for focused search
        self.F = np.random.uniform(0.3, 0.9)  # Extended range for differential weight
        self.CR = np.random.uniform(0.6, 1.0)  # Broader crossover probability
        self.w = 0.4  # Lower inertia for rapid adaptation
        self.c1 = 1.3  # Slightly lower cognitive learning
        self.c2 = 1.7  # Higher social learning for better convergence
        self.velocity_clamp = 0.2 * (self.upper_bound - self.lower_bound)  # Modified velocity clamp
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim)) * self.velocity_clamp
        self.personal_best = np.copy(self.population)
        self.global_best = np.copy(self.population[np.argmin([float('inf')] * self.population_size)])
        self.best_fitness = float('inf')
        self.strategy_dynamic = True  # Keep dynamic strategy selection
        self.adaptive_population = np.random.choice([True, False], p=[0.7, 0.3])  # Adaptive strategy for population

    def __call__(self, func):
        evals = 0
        fitness = np.array([func(ind) for ind in self.population])
        self.personal_fitness = np.copy(fitness)
        evals += self.population_size

        while evals < self.budget:
            if self.strategy_dynamic:  # Dynamic adaptation
                self.F = 0.3 + 0.6 * np.random.rand()  # Extended range adjustment
                self.CR = 0.6 + 0.4 * np.random.rand()  # Broader range for crossover

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

            # Adaptive population strategy
            if self.adaptive_population and evals < self.budget * 0.8:  # Apply only in the initial 80% of evaluations
                best_half_idx = np.argsort(fitness)[:self.population_size // 2]
                self.population[best_half_idx] = self.population[best_half_idx] + np.random.normal(0, 0.1, (self.population_size // 2, self.dim))
                fitness[best_half_idx] = [func(ind) for ind in self.population[best_half_idx]]
                evals += len(best_half_idx)

        return self.global_best, self.best_fitness