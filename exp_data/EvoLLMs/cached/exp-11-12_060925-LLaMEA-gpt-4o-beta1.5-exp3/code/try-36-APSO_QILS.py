import numpy as np

class APSO_QILS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.evaluations = 0
        self.pop_size = 10 * dim
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.quantum_delta = 0.1
        
    def __call__(self, func):
        # Initialize particle positions and velocities
        positions = self.lower_bound + np.random.rand(self.pop_size, self.dim) * (self.upper_bound - self.lower_bound)
        velocities = np.random.rand(self.pop_size, self.dim) * 0.1
        fitness = np.apply_along_axis(func, 1, positions)
        self.evaluations = self.pop_size

        # Initialize personal bests and global best
        personal_best_positions = np.copy(positions)
        personal_best_fitness = np.copy(fitness)
        global_best_idx = np.argmin(fitness)
        global_best_position = positions[global_best_idx]
        global_best_fitness = fitness[global_best_idx]

        while self.evaluations < self.budget:
            # Update velocities and positions
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_coeff * r1 * (personal_best_positions - positions)
                          + self.social_coeff * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            # Apply quantum-inspired local search
            quantum_perturbation = (np.random.rand(self.pop_size, self.dim) - 0.5) * self.quantum_delta
            quantum_positions = np.clip(positions + quantum_perturbation, self.lower_bound, self.upper_bound)
            quantum_fitness = np.apply_along_axis(func, 1, quantum_positions)
            self.evaluations += self.pop_size
            
            # Evaluate fitness and update personal and global bests
            new_fitness = np.apply_along_axis(func, 1, positions)
            self.evaluations += self.pop_size

            for i in range(self.pop_size):
                if new_fitness[i] < personal_best_fitness[i]:
                    personal_best_positions[i] = positions[i]
                    personal_best_fitness[i] = new_fitness[i]

                if quantum_fitness[i] < personal_best_fitness[i]:
                    personal_best_positions[i] = quantum_positions[i]
                    personal_best_fitness[i] = quantum_fitness[i]

            global_best_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[global_best_idx] < global_best_fitness:
                global_best_position = personal_best_positions[global_best_idx]
                global_best_fitness = personal_best_fitness[global_best_idx]

        return global_best_position, global_best_fitness