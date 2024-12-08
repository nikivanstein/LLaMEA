import numpy as np

class EnhancedHyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5 + 0.2 * np.random.rand()  # More dynamic range for Differential weight
        self.CR = 0.8  # Fine-tuned Crossover probability
        self.w = 0.3 + 0.1 * np.random.rand()  # Randomized inertia for variance reduction
        self.c1 = 1.8  # Fine-tuned cognitive learning
        self.c2 = 1.5  # Fine-tuned social learning
        self.velocity_clamp = 0.3 * (self.upper_bound - self.lower_bound)  # Adjusted velocity clamp
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.zeros((self.population_size, dim))  # Start with zero velocities
        self.personal_best = np.copy(self.population)
        self.fitness = np.full(self.population_size, np.inf)
        self.global_best = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        evals = 0
        for i in range(self.population_size):
            self.fitness[i] = func(self.population[i])
            evals += 1
            if self.fitness[i] < self.best_fitness:
                self.global_best = self.population[i]
                self.best_fitness = self.fitness[i]

        while evals < self.budget:
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
                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    self.personal_best[i] = trial if trial_fitness < func(self.personal_best[i]) else self.personal_best[i]

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