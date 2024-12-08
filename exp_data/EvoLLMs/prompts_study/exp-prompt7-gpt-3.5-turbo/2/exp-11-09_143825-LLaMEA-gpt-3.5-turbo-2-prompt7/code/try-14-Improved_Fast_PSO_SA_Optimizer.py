import numpy as np

class Improved_Fast_PSO_SA_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_pos = np.random.uniform(-5.0, 5.0, (budget, dim))
        self.particle_vel = np.random.uniform(-1, 1, (budget, dim))
        self.global_best = np.random.uniform(-5.0, 5.0, dim)
        self.global_best_score = float('inf')
        self.temperature = 1.0
        self.inertia_weight = 0.7
        self.alpha = 0.1

    def __call__(self, func):
        for _ in range(self.budget):
            for i in range(self.budget):
                fitness = func(self.particle_pos[i])
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best = self.particle_pos[i]
                if fitness < func(self.global_best):
                    self.global_best = self.particle_pos[i]

                new_vel = self.inertia_weight * self.particle_vel[i] + np.random.uniform(0, 1) * (self.global_best - self.particle_pos[i]) + np.random.uniform(0, 1) * (self.particle_pos[i] - self.particle_pos[i])
                new_pos = self.particle_pos[i] + new_vel
                new_pos = np.clip(new_pos, -5.0, 5.0)
                new_fitness = func(new_pos)

                if new_fitness < fitness or np.random.rand() < np.exp((fitness - new_fitness) / self.temperature):
                    self.particle_pos[i] = new_pos
                    self.particle_vel[i] = new_vel

            self.temperature *= 0.95  # Annealing schedule
            self.inertia_weight *= 1 - self.alpha

        return self.global_best