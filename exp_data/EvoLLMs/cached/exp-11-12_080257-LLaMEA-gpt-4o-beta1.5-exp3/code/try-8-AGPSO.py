import numpy as np

class AGPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = max(10, 5 * dim)  # Larger swarm for diversity
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.eval_count = 0

    def initialize_swarm(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        return positions, velocities

    def evaluate_swarm(self, positions, func):
        fitness = np.array([func(pos) for pos in positions])
        self.eval_count += len(positions)
        return fitness

    def update_velocity(self, velocities, positions, personal_best_positions, global_best_position):
        r1 = np.random.rand(self.swarm_size, self.dim)
        r2 = np.random.rand(self.swarm_size, self.dim)
        velocities = (self.inertia_weight * velocities +
                      self.cognitive_coeff * r1 * (personal_best_positions - positions) +
                      self.social_coeff * r2 * (global_best_position - positions))
        return velocities

    def update_position(self, positions, velocities):
        new_positions = positions + velocities
        return np.clip(new_positions, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        positions, velocities = self.initialize_swarm()
        fitness = self.evaluate_swarm(positions, func)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)

        global_best_position = positions[np.argmin(fitness)]
        global_best_fitness = np.min(fitness)

        while self.eval_count < self.budget:
            velocities = self.update_velocity(velocities, positions, personal_best_positions, global_best_position)
            positions = self.update_position(positions, velocities)
            
            fitness = self.evaluate_swarm(positions, func)

            # Update personal bests
            better_mask = fitness < personal_best_fitness
            personal_best_positions[better_mask] = positions[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]

            # Update global best
            min_fitness_idx = np.argmin(fitness)
            if fitness[min_fitness_idx] < global_best_fitness:
                global_best_fitness = fitness[min_fitness_idx]
                global_best_position = positions[min_fitness_idx]

        return global_best_position