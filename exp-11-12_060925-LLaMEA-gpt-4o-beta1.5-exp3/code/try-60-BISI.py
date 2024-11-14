import numpy as np

class BISI:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 20 * dim
        self.alpha = 0.5  # Influence of best-known positions
        self.beta = 0.3   # Influence of random movement

    def __call__(self, func):
        swarm = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.swarm_size, self.dim) * 0.1
        fitness = np.apply_along_axis(func, 1, swarm)
        self.evaluations = self.swarm_size

        personal_best_positions = np.copy(swarm)
        personal_best_fitness = np.copy(fitness)

        global_best_idx = np.argmin(fitness)
        global_best_position = swarm[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocities using both personal and global bests
                r1, r2 = np.random.rand(2)
                cognitive_component = self.alpha * r1 * (personal_best_positions[i] - swarm[i])
                social_component = self.alpha * r2 * (global_best_position - swarm[i])
                random_movement = self.beta * (np.random.rand(self.dim) - 0.5)

                velocities[i] = velocities[i] + cognitive_component + social_component + random_movement
                swarm[i] = np.clip(swarm[i] + velocities[i], self.lower_bound, self.upper_bound)

                # Evaluate new position
                new_fitness = func(swarm[i])
                self.evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = swarm[i]
                    personal_best_fitness[i] = new_fitness

                # Update global best
                if new_fitness < global_best_fitness:
                    global_best_position = swarm[i]
                    global_best_fitness = new_fitness

        return global_best_position, global_best_fitness