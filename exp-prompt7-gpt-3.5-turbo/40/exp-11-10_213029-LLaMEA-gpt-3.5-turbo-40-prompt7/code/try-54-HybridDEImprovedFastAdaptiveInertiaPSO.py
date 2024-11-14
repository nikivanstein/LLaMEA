import numpy as np

class HybridDEImprovedFastAdaptiveInertiaPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30
        self.max_vel = 0.2
        self.c1_min, self.c1_max = 1.5, 2.5
        self.c2_min, self.c2_max = 1.5, 2.5
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.position = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.velocity = np.random.uniform(-self.max_vel, self.max_vel, (self.pop_size, self.dim))
        self.personal_best_pos = np.copy(self.position)
        self.personal_best_val = np.full(self.pop_size, np.inf)
        self.global_best_pos = None
        self.global_best_val = np.inf

    def __call__(self, func):
        for _ in range(self.budget):
            fitness = func(self.position)
            for i in range(self.pop_size):
                if fitness[i] < self.personal_best_val[i]:
                    self.personal_best_val[i] = fitness[i]
                    self.personal_best_pos[i] = np.copy(self.position[i])
                if fitness[i] < self.global_best_val:
                    self.global_best_val = fitness[i]
                    self.global_best_pos = np.copy(self.position[i])
            inertia_weight = self.inertia_max - (_ / self.budget) * (self.inertia_max - self.inertia_min)
            c1 = self.c1_min + (_ / self.budget) * (self.c1_max - self.c1_min)
            c2 = self.c2_min + (_ / self.budget) * (self.c2_max - self.c2_min)
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity[i] = inertia_weight * self.velocity[i] + c1 * r1 * (self.personal_best_pos[i] - self.position[i]) + c2 * r2 * (self.global_best_pos - self.position[i])
                self.velocity[i] = np.clip(self.velocity[i], -self.max_vel, self.max_vel)
                self.position[i] += self.velocity[i]
                self.position[i] = np.clip(self.position[i], -5.0, 5.0)
            self.pop_size = max(5, int(30 * (1 - _ / self.budget))) if _ % (self.budget // 20) == 0 else self.pop_size  # Dynamic population adjustment every 5% of the budget
            self.position = np.vstack((self.position[:self.pop_size], np.random.uniform(-5.0, 5.0, (30 - self.pop_size, self.dim))))
            self.velocity = np.vstack((self.velocity[:self.pop_size], np.random.uniform(-self.max_vel, self.max_vel, (30 - self.pop_size, self.dim))))
            self.personal_best_pos = np.vstack((self.personal_best_pos[:self.pop_size], np.copy(self.position[self.pop_size:])))
            self.personal_best_val = np.append(self.personal_best_val[:self.pop_size], np.full(30 - self.pop_size, np.inf))
            if np.random.rand() < 0.2:  # Integrate Differential Evolution for global exploration
                for i in range(self.pop_size):
                    idxs = np.random.choice(range(self.pop_size), 3, replace=False)
                    mutant = self.position[idxs[0]] + 0.5 * (self.position[idxs[1]] - self.position[idxs[2]])
                    cross_points = np.random.rand(self.dim) < 0.9
                    self.position[i] = np.where(cross_points, mutant, self.position[i])
        return self.global_best_pos