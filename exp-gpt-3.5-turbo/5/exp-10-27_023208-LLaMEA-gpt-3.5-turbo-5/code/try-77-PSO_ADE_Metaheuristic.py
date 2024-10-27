import numpy as np

class PSO_ADE_Metaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.max_iter = budget // self.swarm_size
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.cr = 0.5

    def _initialize_swarm(self):
        self.swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))

    def _update_particle(self, target_index, func):
        best_neighbor = self.swarm[np.argmin(np.linalg.norm(self.swarm - self.swarm[target_index], axis=1))]
        r1, r2 = np.random.uniform(0, 1, 2)
        self.velocities[target_index] = self.w * self.velocities[target_index] + self.c1 * r1 * (best_neighbor - self.swarm[target_index]) + self.c2 * r2 * (self.global_best - self.swarm[target_index])
        candidate_position = self.swarm[target_index] + self.velocities[target_index]
        trial_vector = np.copy(self.swarm[target_index])
        for j in range(self.dim):
            if np.random.rand() < self.cr:
                trial_vector[j] = candidate_position[j]
        if func(trial_vector) < func(self.swarm[target_index]):
            self.swarm[target_index] = trial_vector
            if func(trial_vector) < func(self.global_best):
                self.global_best = np.copy(trial_vector)

    def __call__(self, func):
        self.global_best = np.random.uniform(-5.0, 5.0, self.dim)
        self._initialize_swarm()
        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                self._update_particle(i, func)
        return self.global_best

pso_ade = PSO_ADE_Metaheuristic(budget=1000, dim=10)