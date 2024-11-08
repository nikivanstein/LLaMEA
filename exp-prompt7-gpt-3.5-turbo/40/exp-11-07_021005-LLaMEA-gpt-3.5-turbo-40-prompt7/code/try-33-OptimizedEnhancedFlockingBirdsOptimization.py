import numpy as np

class OptimizedEnhancedFlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.pop_size, self.max_vel, self.c1, self.c2, self.w = budget, dim, 20, 0.1, 2.0, 2.0, 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.pop_size, dim))
        self.velocities = np.random.uniform(-self.max_vel, self.max_vel, (self.pop_size, dim))
        self.personal_best_pos, self.personal_best_val = self.positions.copy(), np.full(self.pop_size, np.inf)
        self.global_best_pos, self.global_best_val = np.zeros(dim), np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = np.array([func(p) for p in self.positions])
            update_pb = fitness < self.personal_best_val
            self.personal_best_val[update_pb] = fitness[update_pb]
            self.personal_best_pos[update_pb] = self.positions[update_pb]
            
            gb_mask = fitness < self.global_best_val
            self.global_best_val = np.where(gb_mask, fitness, self.global_best_val)
            self.global_best_pos = np.where(gb_mask[:, None], self.positions, self.global_best_pos)

            r1, r2 = np.random.random((self.pop_size, self.dim)), np.random.random((self.pop_size, self.dim))
            self.velocities = self.w * self.velocities + self.c1 * r1 * (self.personal_best_pos - self.positions) + self.c2 * r2 * (self.global_best_pos - self.positions)
            self.velocities = np.clip(self.velocities, -self.max_vel, self.max_vel)
            self.positions = np.clip(self.positions + self.velocities, -5.0, 5.0)

        return self.global_best_val.max()