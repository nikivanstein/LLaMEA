import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.inertia_weight = 0.5
        self.cognitive_coefficient = 1.5
        self.social_coefficient = 1.0 + 1.0 * np.random.rand()
        self.current_evaluations = 0

    def initialize_population(self):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, 
                                      (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        return positions, velocities

    def evaluate_population(self, positions, func):
        fitness = np.apply_along_axis(func, 1, positions)
        self.current_evaluations += self.population_size
        return fitness

    def update_velocities_and_positions(self, positions, velocities, 
                                        personal_best_positions, global_best_position):
        r_p = np.random.uniform(0, 1, (self.population_size, self.dim))
        r_g = np.random.uniform(0, 1, (self.population_size, self.dim))
        
        # Adjusted cognitive and social coefficients
        self.cognitive_coefficient = 1.5 + 0.5 * (self.current_evaluations / self.budget)
        self.social_coefficient = 1.5 - 0.5 * (self.current_evaluations / self.budget)
        
        cognitive_term = self.cognitive_coefficient * r_p * (personal_best_positions - positions)
        social_term = self.social_coefficient * r_g * (global_best_position - positions)
        
        diversity = np.std(positions, axis=0)
        adaptive_inertia = 0.4 + 0.1 * (diversity / (np.max(diversity) + 1e-5))
        self.inertia_weight = adaptive_inertia * (1 - self.current_evaluations / self.budget)
        
        velocities = self.inertia_weight * velocities + cognitive_term + social_term
        positions += velocities
        positions = np.clip(positions, self.lower_bound, self.upper_bound)
        return positions, velocities

    def mutation_strategy(self, position, best_positions, fitness_variance):
        indices = np.random.choice(self.population_size, 3, replace=False)
        a, b, c = best_positions[indices]
        perturbation_scale = 0.5 + 0.6 * (1 - fitness_variance / (np.max(fitness_variance) + 1e-5))  # Adjusted
        mutant = a + perturbation_scale * (b - c)
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

            fitness_variance = np.var(fitness)
            for i in range(self.population_size):
                if np.random.rand() < 0.1 + 0.2 * (fitness_variance / (np.max(fitness_variance) + 1e-5)):
                    positions[i] = self.mutation_strategy(positions[i], personal_best_positions, fitness_variance)

            fitness = self.evaluate_population(positions, func)
            
            better_indices = fitness < personal_best_fitness
            personal_best_fitness[better_indices] = fitness[better_indices]
            personal_best_positions[better_indices] = positions[better_indices]

            global_best_index = np.argmin(personal_best_fitness)
            global_best_position = personal_best_positions[global_best_index]

        return global_best_position, personal_best_fitness[global_best_index]