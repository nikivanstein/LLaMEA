import numpy as np

class AdaptiveMemeticOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.eval_count = 0
        self.omega = 0.5
        self.phi_p = 0.5
        self.phi_g = 0.5
        self.elite_fraction = 0.1

    def initialize_population(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        return positions, velocities

    def evaluate_population(self, positions, func):
        fitness = np.array([func(ind) for ind in positions])
        self.eval_count += len(positions)
        return fitness

    def update_velocity_position(self, positions, velocities, personal_best_positions, global_best_position):
        r_p = np.random.rand(self.population_size, self.dim)
        r_g = np.random.rand(self.population_size, self.dim)

        velocities = self.omega * velocities + self.phi_p * r_p * (personal_best_positions - positions) + self.phi_g * r_g * (global_best_position - positions)
        positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)
        return positions, velocities

    def local_search(self, best_position, func):
        epsilon = 0.1
        neighbors = np.clip(best_position + epsilon * np.random.uniform(-1, 1, (5, self.dim)), self.lower_bound, self.upper_bound)
        neighbor_fitness = self.evaluate_population(neighbors, func)
        best_idx = np.argmin(neighbor_fitness)
        return neighbors[best_idx], neighbor_fitness[best_idx]

    def __call__(self, func):
        positions, velocities = self.initialize_population()
        fitness = self.evaluate_population(positions, func)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx]

        while self.eval_count < self.budget:
            positions, velocities = self.update_velocity_position(positions, velocities, personal_best_positions, global_best_position)
            fitness = self.evaluate_population(positions, func)

            # Update personal bests
            better_mask = fitness < personal_best_fitness
            personal_best_positions[better_mask] = positions[better_mask]
            personal_best_fitness[better_mask] = fitness[better_mask]

            # Update global best
            current_global_best_idx = np.argmin(fitness)
            if fitness[current_global_best_idx] < personal_best_fitness[global_best_idx]:
                global_best_idx = current_global_best_idx
                global_best_position = positions[global_best_idx]

            # Local search on elite individuals
            num_elites = max(1, int(self.elite_fraction * self.population_size))
            elite_indices = np.argsort(fitness)[:num_elites]
            for idx in elite_indices:
                if self.eval_count >= self.budget:
                    break
                improved, improved_fitness = self.local_search(positions[idx], func)
                if improved_fitness < fitness[idx]:
                    positions[idx] = improved
                    fitness[idx] = improved_fitness

        return positions[global_best_idx]