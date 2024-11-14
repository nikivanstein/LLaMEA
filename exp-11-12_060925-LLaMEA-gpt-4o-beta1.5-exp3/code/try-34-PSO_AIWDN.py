import numpy as np

class PSO_AIWDN:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 10 * dim
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.min_inertia_weight = 0.4
        self.max_inertia_weight = 0.9

    def __call__(self, func):
        positions = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.swarm_size, self.dim) * 0.1 * (self.upper_bound - self.lower_bound)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best_positions)
        self.evaluations += self.swarm_size

        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)

                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.social_coeff * r2 * (global_best_position - positions[i]))

                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)

                fitness = func(positions[i])
                self.evaluations += 1

                if fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = fitness

                    if fitness < global_best_fitness:
                        global_best_position = positions[i]
                        global_best_fitness = fitness

            inertia_decay = (self.max_inertia_weight - self.min_inertia_weight) / self.budget
            self.inertia_weight = max(self.min_inertia_weight, self.inertia_weight - inertia_decay)

        return global_best_position, global_best_fitness