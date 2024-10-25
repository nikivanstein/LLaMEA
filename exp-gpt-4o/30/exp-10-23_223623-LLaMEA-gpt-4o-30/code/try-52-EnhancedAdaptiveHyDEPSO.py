import numpy as np

class EnhancedAdaptiveHyDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 12 * dim  # Slightly increased population size
        self.F = 0.5 + 0.3 * np.random.rand()  # Self-adaptive Differential weight
        self.CR = 0.8  # Adjusted Crossover probability
        self.w = 0.5  # Increased Inertia weight for better balance
        self.c1 = 1.4  # Balanced cognitive learning
        self.c2 = 1.6  # Increased social learning factor
        self.velocity_clamp = 0.3 * (self.upper_bound - self.lower_bound)  # Increased velocity clamp
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim)) * self.velocity_clamp
        self.personal_best = np.copy(self.population)
        self.global_best = np.copy(self.population[np.argmin([float('inf')] * self.population_size)])
        self.best_fitness = float('inf')
        self.strategy_dynamic = True  # Enable dynamic strategy selection
        self.triple_agent_strategy = True  # Introduce triple agent strategy

    def __call__(self, func):
        evals = 0
        fitness = np.array([func(ind) for ind in self.population])
        evals += self.population_size

        while evals < self.budget:
            if self.strategy_dynamic:  # Dynamic adaptation
                self.F = 0.3 + 0.3 * np.random.rand()  # Randomize within a range
                self.CR = 0.75 + 0.2 * np.random.rand()  # Randomize within a range

            agents = np.random.choice(['DE', 'PSO', 'Hybrid'], size=self.population_size, p=[0.3, 0.3, 0.4])
            for i in range(self.population_size):
                if agents[i] == 'DE':
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[indices]
                    mutant = np.clip(x_r1 + self.F * (x_r2 - x_r3), self.lower_bound, self.upper_bound)
                    crossover = np.random.rand(self.dim) < self.CR
                    trial = np.where(crossover, mutant, self.population[i])
                elif agents[i] == 'PSO':
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive = self.c1 * r1 * (self.personal_best[i] - self.population[i])
                    social = self.c2 * r2 * (self.global_best - self.population[i])
                    self.velocities[i] = self.w * self.velocities[i] + cognitive + social
                    self.velocities[i] = np.clip(self.velocities[i], -self.velocity_clamp, self.velocity_clamp)
                    trial = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)
                else:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    x_r1, x_r2, x_r3 = self.population[indices]
                    mutant = np.clip(x_r1 + self.F * (x_r2 - x_r3), self.lower_bound, self.upper_bound)
                    crossover = np.random.rand(self.dim) < self.CR
                    trial = np.where(crossover, mutant, self.population[i])
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive = self.c1 * r1 * (self.personal_best[i] - trial)
                    social = self.c2 * r2 * (self.global_best - trial)
                    trial_velocity = self.w * self.velocities[i] + cognitive + social
                    self.velocities[i] = np.clip(trial_velocity, -self.velocity_clamp, self.velocity_clamp)
                    trial = np.clip(trial + self.velocities[i], self.lower_bound, self.upper_bound)
                    
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness[i]:
                    self.population[i] = trial
                    fitness[i] = trial_fitness
                    self.personal_best[i] = trial if trial_fitness < func(self.personal_best[i]) else self.personal_best[i]

                if trial_fitness < self.best_fitness:
                    self.global_best = trial
                    self.best_fitness = trial_fitness

        return self.global_best, self.best_fitness