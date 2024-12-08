import numpy as np

class QuantumInspiredParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, float('inf'))
        self.func_evaluations = 0
        self.global_best_score = float('inf')
        self.global_best_position = None
        self.c1 = 2.0  # cognitive coefficient
        self.c2 = 2.0  # social coefficient
        self.w = 0.7   # inertia weight
        self.quantum_factor = 0.05  # Quantum behavior factor

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate the fitness
                fitness = func(self.positions[i])
                self.func_evaluations += 1

                # Update personal best
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            
            cognitive_component = self.c1 * r1 * (self.personal_best_positions - self.positions)
            social_component = self.c2 * r2 * (self.global_best_position - self.positions)
            
            self.velocities = self.w * self.velocities + cognitive_component + social_component

            # Quantum-inspired position update
            self.positions += self.velocities + self.quantum_factor * np.random.normal(0, 1, (self.population_size, self.dim))
            
            # Apply bounds
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        return self.global_best_position