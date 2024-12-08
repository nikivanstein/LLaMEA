import numpy as np

class MutativeInertiaPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 30  # Adjust population size dynamically based on convergence
        self.max_vel = 0.2
        self.c1 = 2.0
        self.c2 = 2.0
        self.inertia_min = 0.4
        self.inertia_max = 0.9
        self.mutation_rate = 0.05  # Introduce mutation for exploration
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
            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.velocity[i] = inertia_weight * self.velocity[i] + self.c1 * r1 * (self.personal_best_pos[i] - self.position[i]) + self.c2 * r2 * (self.global_best_pos - self.position[i])
                self.velocity[i] = np.clip(self.velocity[i], -self.max_vel, self.max_vel)
                self.position[i] += self.velocity[i]
                self.position[i] = np.clip(self.position[i], -5.0, 5.0)
                # Introduce mutation for exploration
                if np.random.rand() < self.mutation_rate:
                    self.position[i] += np.random.uniform(-0.5, 0.5, self.dim)
                    self.position[i] = np.clip(self.position[i], -5.0, 5.0)
            # Dynamic population size adjustment based on convergence
            self.pop_size = max(5, int(30 * (1 - _ / self.budget)))
            self.position = np.vstack((self.position[:self.pop_size], np.random.uniform(-5.0, 5.0, (30 - self.pop_size, self.dim))))
            self.velocity = np.vstack((self.velocity[:self.pop_size], np.random.uniform(-self.max_vel, self.max_vel, (30 - self.pop_size, self.dim))))
            self.personal_best_pos = np.vstack((self.personal_best_pos[:self.pop_size], np.copy(self.position[self.pop_size:])))
            self.personal_best_val = np.append(self.personal_best_val[:self.pop_size], np.full(30 - self.pop_size, np.inf))
        return self.global_best_pos