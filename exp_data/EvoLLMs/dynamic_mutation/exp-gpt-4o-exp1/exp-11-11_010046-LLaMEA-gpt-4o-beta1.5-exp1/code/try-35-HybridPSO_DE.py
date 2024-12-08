import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight_min = 0.4
        self.inertia_weight_max = 0.9
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.0 + 1.0 * np.random.rand()
        self.current_evaluations = 0

    def initialize_population(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        return positions, velocities

    def evaluate_population(self, positions, func):
        fitness = np.apply_along_axis(func, 1, positions)
        self.current_evaluations += self.population_size
        return fitness

    def update_velocities_and_positions(self, positions, velocities, personal_best_positions, global_best_position):
        r_p = np.random.uniform(0, 1, (self.population_size, self.dim))
        r_g = np.random.uniform(0, 1, (self.population_size, self.dim))
        inertia_weight = self.inertia_weight_max - (self.inertia_weight_max - self.inertia_weight_min) * (self.current_evaluations / self.budget)
        cognitive_term = self.cognitive_coefficient * r_p * (personal_best_positions - positions)
        social_term = self.social_coefficient * r_g * (global_best_position - positions)
        velocities = inertia_weight * velocities + cognitive_term + social_term
        positions += velocities
        positions = np.clip(positions, self.lower_bound, self.upper_bound)
        return positions, velocities

    def mutation_strategy(self, position, neighborhood_positions):
        indices = np.random.choice(len(neighborhood_positions), 3, replace=False)
        a, b, c = neighborhood_positions[indices]
        mutant = a + 0.8 * (b - c)
        return np.clip(mutant, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        positions, velocities = self.initialize_population()
        fitness = self.evaluate_population(positions, func)
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        global_best_index = np.argmin(fitness)
        global_best_position = positions[global_best_index]

        while self.current_evaluations < self.budget:
            positions, velocities = self.update_velocities_and_positions(
                positions, velocities, personal_best_positions, global_best_position)

            mutation_probability = 0.2 * (1 - self.current_evaluations / self.budget)
            for i in range(self.population_size):
                if np.random.rand() < mutation_probability:
                    neighborhood_indices = np.random.choice(self.population_size, 5, replace=False)
                    neighborhood_positions = personal_best_positions[neighborhood_indices]
                    positions[i] = self.mutation_strategy(positions[i], neighborhood_positions)

            fitness = self.evaluate_population(positions, func)
            
            better_indices = fitness < personal_best_fitness
            personal_best_fitness[better_indices] = fitness[better_indices]
            personal_best_positions[better_indices] = positions[better_indices]

            global_best_index = np.argmin(personal_best_fitness)
            global_best_position = personal_best_positions[global_best_index]

        return global_best_position, personal_best_fitness[global_best_index]