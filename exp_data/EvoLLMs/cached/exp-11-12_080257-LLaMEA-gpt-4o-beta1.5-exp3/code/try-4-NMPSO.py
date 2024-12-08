import numpy as np

class NMPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(5, 3 * dim)
        self.c1 = 1.5  # Cognitive component
        self.c2 = 1.5  # Social component
        self.w = 0.7  # Inertia weight
        self.eval_count = 0
        self.velocity_clamp = (self.lower_bound - self.upper_bound) * 0.2

    def initialize_population(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(self.velocity_clamp), abs(self.velocity_clamp), (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        return positions, velocities, personal_best_positions

    def evaluate_population(self, positions, func):
        fitness = np.array([func(pos) for pos in positions])
        self.eval_count += len(positions)
        return fitness

    def update_velocity(self, velocities, positions, personal_best_positions, global_best_position):
        r1, r2 = np.random.rand(2, self.population_size, self.dim)
        cognitive_velocity = self.c1 * r1 * (personal_best_positions - positions)
        social_velocity = self.c2 * r2 * (global_best_position - positions)
        new_velocities = self.w * velocities + cognitive_velocity + social_velocity
        return np.clip(new_velocities, -self.velocity_clamp, self.velocity_clamp)

    def update_position(self, positions, velocities):
        new_positions = positions + velocities
        return np.clip(new_positions, self.lower_bound, self.upper_bound)

    def local_search(self, position, func):
        epsilon = 0.05
        neighbors = np.clip(position + epsilon * np.random.uniform(-1, 1, (5, self.dim)), self.lower_bound, self.upper_bound)
        neighbor_fitness = self.evaluate_population(neighbors, func)
        best_neighbor_idx = np.argmin(neighbor_fitness)
        return neighbors[best_neighbor_idx], neighbor_fitness[best_neighbor_idx]

    def __call__(self, func):
        positions, velocities, personal_best_positions = self.initialize_population()
        personal_best_fitness = self.evaluate_population(personal_best_positions, func)
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]

        while self.eval_count < self.budget:
            velocities = self.update_velocity(velocities, positions, personal_best_positions, global_best_position)
            positions = self.update_position(positions, velocities)
            fitness = self.evaluate_population(positions, func)

            # Update personal bests
            improved = fitness < personal_best_fitness
            personal_best_positions[improved] = positions[improved]
            personal_best_fitness[improved] = fitness[improved]

            # Update global best
            current_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_best_idx] < personal_best_fitness[global_best_idx]:
                global_best_idx = current_best_idx
                global_best_position = personal_best_positions[global_best_idx]

            # Perform local search on the global best
            if self.eval_count < self.budget:
                improved_position, improved_fitness = self.local_search(global_best_position, func)
                if improved_fitness < personal_best_fitness[global_best_idx]:
                    personal_best_positions[global_best_idx] = improved_position
                    personal_best_fitness[global_best_idx] = improved_fitness
                    global_best_position = improved_position

        return global_best_position