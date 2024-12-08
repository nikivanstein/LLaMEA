import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim, pop_size=50, omega=0.5, phi_p=0.5, phi_g=0.5):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.omega = omega
        self.phi_p = phi_p
        self.phi_g = phi_g

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
            self.velocities = inertia_weight * self.velocities \
                              + self.phi_p * np.random.rand(self.pop_size, self.dim) * (self.personal_best_positions - self.positions) \
                              + self.phi_g * np.random.rand(self.pop_size, self.dim) * (self.global_best_position - self.positions)

            self.positions += self.velocities

            # Ensure particles do not exceed the search space boundaries
            self.positions = np.clip(self.positions, -5.0, 5.0)

        return self.global_best_position, self.global_best_score