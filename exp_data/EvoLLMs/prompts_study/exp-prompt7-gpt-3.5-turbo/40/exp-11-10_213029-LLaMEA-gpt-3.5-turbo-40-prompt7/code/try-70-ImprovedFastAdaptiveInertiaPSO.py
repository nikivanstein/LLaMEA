import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ImprovedFastAdaptiveInertiaPSO:
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
        with ThreadPoolExecutor() as executor:
            for _ in range(self.budget):
                fitness = func(self.position)
                futures = [executor.submit(self.update_particle, i, fitness, func) for i in range(self.pop_size)]
                for future in futures:
                    result = future.result()
                    i, new_fitness = result
                    if new_fitness < self.global_best_val:
                        self.global_best_val = new_fitness
                        self.global_best_pos = np.copy(self.position[i])
            return self.global_best_pos

    def update_particle(self, i, fitness, func):
        new_fitness = fitness[i]
        if new_fitness < self.personal_best_val[i]:
            self.personal_best_val[i] = new_fitness
            self.personal_best_pos[i] = np.copy(self.position[i])
        inertia_weight = self.inertia_max - (_ / self.budget) * (self.inertia_max - self.inertia_min)
        c1 = self.c1_min + (_ / self.budget) * (self.c1_max - self.c1_min)
        c2 = self.c2_min + (_ / self.budget) * (self.c2_max - self.c2_min)
        r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
        self.velocity[i] = inertia_weight * self.velocity[i] + c1 * r1 * (self.personal_best_pos[i] - self.position[i]) + c2 * r2 * (self.global_best_pos - self.position[i])
        self.velocity[i] = np.clip(self.velocity[i], -self.max_vel, self.max_vel)
        self.position[i] += self.velocity[i]
        self.position[i] = np.clip(self.position[i], -5.0, 5.0)
        return i, new_fitness