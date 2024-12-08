import numpy as np

class AdaptiveQuantumSwarmOptimization:
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
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.func_evaluations = 0
        self.quantum_probability = 0.1
        self.adaptive_neighbors = 3

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            scores = np.array([func(pos) for pos in self.positions])
            self.func_evaluations += self.population_size

            # Update personal and global bests
            better_scores = scores < self.personal_best_scores
            self.personal_best_scores[better_scores] = scores[better_scores]
            self.personal_best_positions[better_scores] = self.positions[better_scores]

            if np.min(scores) < self.global_best_score:
                self.global_best_score = np.min(scores)
                self.global_best_position = self.positions[np.argmin(scores)]

            # Adaptive neighborhood topology
            neighbors = np.random.choice(self.population_size, self.adaptive_neighbors, replace=False)
            local_best_position = self.positions[neighbors[np.argmin(scores[neighbors])]]

            for i in range(self.population_size):
                quantum_move = np.random.rand() < self.quantum_probability
                if quantum_move:
                    self.velocities[i] = np.random.normal(0, 0.1, self.dim)
                else:
                    cognitive_component = np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
                    social_component = np.random.rand(self.dim) * (local_best_position - self.positions[i])
                    self.velocities[i] = 0.5 * self.velocities[i] + cognitive_component + social_component

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            self.quantum_probability = 0.1 * (1 - np.cos(2 * np.pi * self.func_evaluations / self.budget))

        return self.global_best_position