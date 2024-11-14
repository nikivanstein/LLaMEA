import numpy as np

class CooperativeSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.swarm_size = 20 * dim
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5

    def __call__(self, func):
        # Initialize the swarm
        swarm_positions = self.lower_bound + np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound)
        swarm_velocities = np.random.rand(self.swarm_size, self.dim) * (self.upper_bound - self.lower_bound) * 0.1
        personal_best_positions = np.copy(swarm_positions)
        personal_best_fitness = np.apply_along_axis(func, 1, personal_best_positions)
        self.evaluations = self.swarm_size

        # Global best initialization
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocities and positions
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm_velocities[i] = (self.inertia_weight * swarm_velocities[i] +
                                       self.cognitive_coef * r1 * (personal_best_positions[i] - swarm_positions[i]) +
                                       self.social_coef * r2 * (global_best_position - swarm_positions[i]))
                swarm_positions[i] += swarm_velocities[i]
                swarm_positions[i] = np.clip(swarm_positions[i], self.lower_bound, self.upper_bound)

                # Evaluate new fitness
                fitness_value = func(swarm_positions[i])
                self.evaluations += 1

                # Update personal best
                if fitness_value < personal_best_fitness[i]:
                    personal_best_positions[i] = swarm_positions[i]
                    personal_best_fitness[i] = fitness_value

                    # Update global best
                    if fitness_value < global_best_fitness:
                        global_best_position = swarm_positions[i]
                        global_best_fitness = fitness_value

            # Dynamic role adaptation: adjust parameters
            if self.evaluations % (self.swarm_size // 2) == 0:
                self.inertia_weight *= 0.95
                self.cognitive_coef = 1.5 + 0.5 * np.random.rand()
                self.social_coef = 1.5 + 0.5 * np.random.rand()
                # Shuffle the swarm to encourage diversity
                np.random.shuffle(swarm_positions)

        return global_best_position, global_best_fitness