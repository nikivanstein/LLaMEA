import numpy as np

class FastAdaptiveSinusoidalPSO(ImprovedFastAdaptiveInertiaPSO):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)
    
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
            self.pop_size = max(5, int(30 * (1 - np.sin(_/self.budget*np.pi))**2)) if _ % (self.budget // 20) == 0 else self.pop_size
            self.position = np.vstack((self.position[:self.pop_size], np.random.uniform(-5.0, 5.0, (30 - self.pop_size, self.dim))))
            self.velocity = np.vstack((self.velocity[:self.pop_size], np.random.uniform(-self.max_vel, self.max_vel, (30 - self.pop_size, self.dim))))
            self.personal_best_pos = np.vstack((self.personal_best_pos[:self.pop_size], np.copy(self.position[self.pop_size:])))
            self.personal_best_val = np.append(self.personal_best_val[:self.pop_size], np.full(30 - self.pop_size, np.inf))
        return self.global_best_pos