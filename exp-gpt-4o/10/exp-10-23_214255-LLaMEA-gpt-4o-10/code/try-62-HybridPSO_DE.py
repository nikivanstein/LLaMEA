import numpy as np

class HybridPSO_DE:
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

        self.global_best_position = np.copy(self.positions[0])
        self.global_best_score = np.inf

        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate and update personal bests
            fitness = np.apply_along_axis(func, 1, self.positions)
            self.evaluations += self.pop_size

            better_mask = fitness < self.personal_best_scores
            self.personal_best_scores[better_mask] = fitness[better_mask]
            self.personal_best_positions[better_mask] = self.positions[better_mask]

            # Update global best
            if np.min(fitness) < self.global_best_score:
                self.global_best_score = np.min(fitness)
                self.global_best_position = np.copy(self.positions[np.argmin(fitness)])

            # Update velocities and positions
            inertia_weight = self.omega * (1 - (self.evaluations / self.budget))
            self.velocities = inertia_weight * self.velocities \
                + self.phi_p * np.random.rand(self.pop_size, self.dim) * (self.personal_best_positions - self.positions) \
                + self.phi_g * np.random.rand(self.pop_size, self.dim) * (self.global_best_position - self.positions)

            self.positions += self.velocities
            self.positions = np.clip(self.positions, -5.0, 5.0)

            # Differential Evolution Mutation and Crossover
            for i in range(self.pop_size):
                indices = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                mutant = self.positions[indices[0]] + self.F * (self.positions[indices[1]] - self.positions[indices[2]])
                mutant = np.clip(mutant, -5.0, 5.0)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.positions[i])
                trial_score = func(trial)
                self.evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial
                    if trial_score < self.global_best_score:
                        self.global_best_score = trial_score
                        self.global_best_position = np.copy(trial)

        return self.global_best_position, self.global_best_score