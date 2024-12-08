import numpy as np

class EnhancedHybridPSO_DE:
    def __init__(self, budget, dim, pop_size=50, omega=0.5, phi_p=0.5, phi_g=0.5, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g
        self.F = F
        self.CR = CR

        self.positions = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, np.inf)

        self.global_best_position = None
        self.global_best_score = np.inf

        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            fitness = np.apply_along_axis(func, 1, self.positions)
            self.evaluations += self.pop_size

            better_mask = fitness < self.personal_best_scores
            self.personal_best_scores[better_mask] = fitness[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            if np.min(fitness) < self.global_best_score:
                self.global_best_score = np.min(fitness)
                self.global_best_position = self.positions[np.argmin(fitness)]

            inertia_weight = self.omega * (1 - (self.evaluations / self.budget))
            dynamic_phi_g = self.phi_g * (np.random.rand() if self.evaluations < self.budget / 2 else 1)
            self.velocities = inertia_weight * self.velocities \
                              + self.phi_p * np.random.rand(self.pop_size, self.dim) * (self.personal_best_positions - self.positions) \
                              + dynamic_phi_g * np.random.rand(self.pop_size, self.dim) * (self.global_best_position - self.positions)

            self.positions += self.velocities

            # Differential Evolution Mutation and Crossover
            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                mutant = self.positions[indices[0]] + self.F * (self.positions[indices[1]] - self.positions[indices[2]])
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.positions[i])
                trial = np.clip(trial, -5.0, 5.0)
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < fitness[i]:
                    self.positions[i] = trial
                    fitness[i] = trial_score
                    if trial_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = trial_score
                        self.personal_best_positions[i] = trial

            # Ensure particles do not exceed the search space boundaries
            self.positions = np.clip(self.positions, -5.0, 5.0)

        return self.global_best_position, self.global_best_score