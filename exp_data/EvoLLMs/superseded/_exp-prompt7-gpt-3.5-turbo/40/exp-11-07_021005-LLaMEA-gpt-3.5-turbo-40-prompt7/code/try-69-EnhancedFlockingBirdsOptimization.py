import numpy as np

class EnhancedFlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size, self.max_vel, self.c1, self.c2, self.w = budget, dim, 20, 0.1, 2.0, 2.0, 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-self.max_vel, self.max_vel, (self.pop_size, self.dim))
        self.personal_best_pos, self.personal_best_vals, self.global_best_pos, self.global_best_val = self.positions.copy(), np.full(self.pop_size, np.inf), np.zeros(self.dim), np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.pop_size):
                fitness = func(self.positions[i])
                if fitness < self.personal_best_vals[i]:
                    self.personal_best_vals[i], self.personal_best_pos[i] = fitness, self.positions[i]
                if fitness < self.global_best_val:
                    self.global_best_val, self.global_best_pos = fitness, self.positions[i]

                r1, r2 = np.random.random(self.dim), np.random.random(self.dim)
                self.velocities[i] = self.w * self.velocities[i] + self.c1 * r1 * (self.personal_best_pos[i] - self.positions[i]) + self.c2 * r2 * (self.global_best_pos - self.positions[i])
                self.velocities[i] = np.clip(self.velocities[i], -self.max_vel, self.max_vel)
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], -5.0, 5.0)

        return self.global_best_val