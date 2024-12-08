import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.9  # Adaptively adjusted
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
        cognitive_term = self.cognitive_coefficient * r_p * (personal_best_positions - positions)
        social_term = self.social_coefficient * r_g * (global_best_position - positions)
        
        # Adaptive inertia weight
        self.inertia_weight = 0.4 + 0.5 * (1 - self.current_evaluations / self.budget)
        
        velocities = self.inertia_weight * velocities + cognitive_term + social_term
        positions += velocities
        positions = np.clip(positions, self.lower_bound, self.upper_bound)
        return positions, velocities

    def mutation_strategy(self, position, best_positions):
        indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = best_positions[indices]
        mutant = a + 0.6 * (b - c)
        
        # Multi-point mutation
        additional_mutation = np.random.choice([a, b, c], size=self.dim)
        mutant = 0.5 * (mutant + additional_mutation)
        
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

            mutation_probability = 0.3 * (1 - self.current_evaluations / self.budget)  # Adjusted probability
            for i in range(self.population_size):
                if np.random.rand() < mutation_probability:
                    positions[i] = self.mutation_strategy(positions[i], personal_best_positions)

            fitness = self.evaluate_population(positions, func)

            better_indices = fitness < personal_best_fitness
            personal_best_fitness[better_indices] = fitness[better_indices]
            personal_best_positions[better_indices] = positions[better_indices]

            global_best_index = np.argmin(personal_best_fitness)
            global_best_position = personal_best_positions[global_best_index]

        return global_best_position, personal_best_fitness[global_best_index]