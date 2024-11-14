import numpy as np

class EnhancedFlockingBirdsOptimization:
    def __init__(self, budget, dim):
        self.budget, self.dim, self.population_size, self.max_velocity, self.c1, self.c2, self.w = budget, dim, 20, 0.1, 2.0, 2.0, 0.5
        self.positions = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-self.max_velocity, self.max_velocity, (self.population_size, self.dim))
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_value = np.inf
        self.personal_best_positions = self.positions.copy()
        self.global_best_position = np.zeros(self.dim)

    def __call__(self, func):
        for _ in range(self.budget):
            fitness_values = np.apply_along_axis(func, 1, self.positions)
            update_personal = fitness_values < self.personal_best_values
            self.personal_best_values[update_personal] = fitness_values[update_personal]
            self.personal_best_positions[update_personal] = self.positions[update_personal]
            
            best_index = np.argmin(fitness_values)
            if fitness_values[best_index] < self.global_best_value:
                self.global_best_value = fitness_values[best_index]
                self.global_best_position = self.positions[best_index]
            
            r1, r2 = np.random.random((self.population_size, self.dim)), np.random.random((self.population_size, self.dim))
            self.velocities = self.w * self.velocities + self.c1 * r1 * (self.personal_best_positions - self.positions) + self.c2 * r2 * (self.global_best_position - self.positions)
            self.velocities = np.clip(self.velocities, -self.max_velocity, self.max_velocity)
            self.positions = np.clip(self.positions + self.velocities, -5.0, 5.0)
        
        return self.global_best_value