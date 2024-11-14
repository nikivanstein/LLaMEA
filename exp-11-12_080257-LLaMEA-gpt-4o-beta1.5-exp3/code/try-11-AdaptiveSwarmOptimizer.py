import numpy as np

class AdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = max(10, 5 * dim)  # Large swarm for better exploration
        self.eval_count = 0
        self.inertia_weight = 0.729  # Constriction factor for convergence
        self.cognitive_const = 1.494
        self.social_const = 1.494

    def initialize_swarm(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        return positions, velocities

    def evaluate_swarm(self, positions, func):
        fitness = np.array([func(pos) for pos in positions])
        self.eval_count += len(positions)
        return fitness

    def update_velocity(self, velocities, positions, personal_best, global_best):
        r1, r2 = np.random.rand(2)
        cognitive = self.cognitive_const * r1 * (personal_best - positions)
        social = self.social_const * r2 * (global_best - positions)
        new_velocities = self.inertia_weight * velocities + cognitive + social
        return np.clip(new_velocities, -1, 1)

    def update_position(self, positions, velocities):
        new_positions = positions + velocities
        return np.clip(new_positions, self.lower_bound, self.upper_bound)

    def local_intensification(self, global_best, func):
        # Simple local search to exploit around the global best
        epsilon = 0.1
        neighbors = np.clip(global_best + epsilon * np.random.uniform(-1, 1, (3, self.dim)), self.lower_bound, self.upper_bound)
        neighbor_fitness = self.evaluate_swarm(neighbors, func)
        best_idx = np.argmin(neighbor_fitness)
        return neighbors[best_idx], neighbor_fitness[best_idx]

    def __call__(self, func):
        positions, velocities = self.initialize_swarm()
        fitness = self.evaluate_swarm(positions, func)
        personal_best = positions.copy()
        personal_best_fitness = fitness.copy()
        global_best_idx = np.argmin(fitness)
        global_best = positions[global_best_idx].copy()

        while self.eval_count < self.budget:
            velocities = self.update_velocity(velocities, positions, personal_best, global_best)
            positions = self.update_position(positions, velocities)
            fitness = self.evaluate_swarm(positions, func)

            better_mask = fitness < personal_best_fitness
            personal_best[better_mask] = positions[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]

            global_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[global_best_idx] < func(global_best):
                global_best = personal_best[global_best_idx].copy()

            # Adaptive inertia weight adjustment
            self.inertia_weight = 0.4 + 0.5 * (self.budget - self.eval_count) / self.budget

            # Local search around the current global best
            if self.eval_count < self.budget:
                improved, improved_fitness = self.local_intensification(global_best, func)
                if improved_fitness < func(global_best):
                    global_best = improved.copy()

        return global_best