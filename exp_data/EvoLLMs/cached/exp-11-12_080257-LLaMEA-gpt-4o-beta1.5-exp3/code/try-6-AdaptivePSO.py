import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = max(10, 5 * dim)  # Larger swarm for better coverage
        self.eval_count = 0
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.neighborhood_size = 3  # Dynamic neighborhood size
        self.local_search_prob = 0.1  # Probability of local restarts

    def initialize_swarm(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        return positions, velocities

    def evaluate_swarm(self, positions, func):
        fitness = np.array([func(pos) for pos in positions])
        self.eval_count += len(positions)
        return fitness

    def update_velocity(self, velocities, positions, personal_best_pos, global_best_pos):
        r1, r2 = np.random.rand(2, self.swarm_size, self.dim)
        cognitive = self.cognitive_coeff * r1 * (personal_best_pos - positions)
        social = self.social_coeff * r2 * (global_best_pos - positions)
        velocities = self.inertia_weight * velocities + cognitive + social
        return np.clip(velocities, -5, 5)

    def local_random_restart(self, positions, personal_best_pos):
        random_restart = np.random.rand(self.swarm_size) < self.local_search_prob
        new_positions = np.where(random_restart[:, np.newaxis], np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim)), positions)
        return np.where(random_restart[:, np.newaxis], new_positions, personal_best_pos)

    def __call__(self, func):
        positions, velocities = self.initialize_swarm()
        fitness = self.evaluate_swarm(positions, func)
        personal_best_pos = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(fitness)
        global_best_pos = positions[global_best_idx]

        while self.eval_count < self.budget:
            velocities = self.update_velocity(velocities, positions, personal_best_pos, global_best_pos)
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            fitness = self.evaluate_swarm(positions, func)
            
            # Update personal bests
            improved_mask = fitness < personal_best_fitness
            personal_best_pos[improved_mask] = positions[improved_mask]
            personal_best_fitness[improved_mask] = fitness[improved_mask]

            # Update global best
            current_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[current_best_idx] < personal_best_fitness[global_best_idx]:
                global_best_idx = current_best_idx
                global_best_pos = personal_best_pos[global_best_idx]

            # Local random restarts for diversity
            positions = self.local_random_restart(positions, personal_best_pos)

        return global_best_pos