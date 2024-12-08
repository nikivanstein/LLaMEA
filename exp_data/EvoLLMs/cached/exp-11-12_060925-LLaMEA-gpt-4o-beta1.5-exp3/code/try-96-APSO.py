import numpy as np

class APSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 20 * dim
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        swarm_position = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        swarm_velocity = np.zeros((self.swarm_size, self.dim))
        fitness = np.apply_along_axis(func, 1, swarm_position)
        self.evaluations = self.swarm_size

        personal_best_position = np.copy(swarm_position)
        personal_best_fitness = np.copy(fitness)

        global_best_idx = np.argmin(fitness)
        global_best_position = np.copy(swarm_position[global_best_idx])
        global_best_fitness = fitness[global_best_idx]

        while self.evaluations < self.budget:
            w = self.w_max - (self.evaluations / self.budget) * (self.w_max - self.w_min)
            
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm_velocity[i] = (w * swarm_velocity[i] +
                                     self.c1 * r1 * (personal_best_position[i] - swarm_position[i]) +
                                     self.c2 * r2 * (global_best_position - swarm_position[i]))
                
                swarm_position[i] += swarm_velocity[i]
                swarm_position[i] = np.clip(swarm_position[i], self.lower_bound, self.upper_bound)

                current_fitness = func(swarm_position[i])
                self.evaluations += 1

                if current_fitness < personal_best_fitness[i]:
                    personal_best_position[i] = swarm_position[i]
                    personal_best_fitness[i] = current_fitness

                    if current_fitness < global_best_fitness:
                        global_best_position = swarm_position[i]
                        global_best_fitness = current_fitness

        return global_best_position, global_best_fitness