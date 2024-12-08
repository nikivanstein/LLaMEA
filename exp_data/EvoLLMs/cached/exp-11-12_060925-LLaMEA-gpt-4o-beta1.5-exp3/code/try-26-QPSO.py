import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.population_size = 10 * dim
        self.inertia_weight = 0.7
        self.personal_coeff = 1.5
        self.global_coeff = 1.5

    def __call__(self, func):
        # Initialize positions and velocities
        positions = self.lower_bound + np.random.rand(self.population_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.population_size, self.dim) * 0.1
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.apply_along_axis(func, 1, positions)
        
        self.evaluations = self.population_size
        global_best_idx = np.argmin(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]

        while self.evaluations < self.budget:
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break

                # Update velocity and position
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                velocities[i] = (self.inertia_weight * velocities[i] +
                                 self.personal_coeff * r1 * (personal_best_positions[i] - positions[i]) +
                                 self.global_coeff * r2 * (global_best_position - positions[i]))
                
                # Quantum-inspired update
                quantum_prob = 0.5 + np.random.rand(self.dim) * 0.1
                quantum_bit = np.where(np.random.rand(self.dim) < quantum_prob, 1, -1)
                new_position = positions[i] + velocities[i] * quantum_bit
                new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                
                # Evaluate new position
                new_fitness = func(new_position)
                self.evaluations += 1

                # Update personal best
                if new_fitness < personal_best_fitness[i]:
                    personal_best_positions[i] = new_position
                    personal_best_fitness[i] = new_fitness

                # Update global best
                if new_fitness < global_best_fitness:
                    global_best_position = new_position
                    global_best_fitness = new_fitness

        return global_best_position, global_best_fitness