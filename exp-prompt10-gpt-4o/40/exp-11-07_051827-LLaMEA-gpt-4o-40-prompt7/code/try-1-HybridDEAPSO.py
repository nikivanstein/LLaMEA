import numpy as np

class HybridDEAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = min(50, budget // 10)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.pop_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_values = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evals = 0
        self.F = 0.5
        self.CR = 0.9
        self.omega = 0.7
        self.phi_p = 1.5
        self.phi_g = 1.5

    def __call__(self, func):
        while self.evals < self.budget:
            if self.global_best_position is None:
                self.evaluate_population(func)

            # Differential Evolution mutation and crossover
            for i in range(self.pop_size):
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial[crossover_mask] = mutant[crossover_mask]
                
                trial_value = func(trial)
                self.evals += 1
                if trial_value < self.personal_best_values[i]:
                    self.personal_best_positions[i] = trial
                    self.personal_best_values[i] = trial_value
                    if trial_value < self.global_best_value:
                        self.global_best_position = trial
                        self.global_best_value = trial_value

            # Particle Swarm Optimization velocity and position update
            r_p = np.random.rand(self.pop_size, self.dim)
            r_g = np.random.rand(self.pop_size, self.dim)
            self.velocities = (self.omega * self.velocities + 
                               self.phi_p * r_p * (self.personal_best_positions - self.population) +
                               self.phi_g * r_g * (self.global_best_position - self.population))
            self.population = np.clip(self.population + self.velocities, self.lower_bound, self.upper_bound)

            # Re-evaluate population
            self.evaluate_population(func)

        return self.global_best_value

    def evaluate_population(self, func):
        for i in range(self.pop_size):
            value = func(self.population[i])
            self.evals += 1
            if value < self.personal_best_values[i]:
                self.personal_best_positions[i] = self.population[i]
                self.personal_best_values[i] = value
                if value < self.global_best_value:
                    self.global_best_value = value
                    self.global_best_position = self.population[i]