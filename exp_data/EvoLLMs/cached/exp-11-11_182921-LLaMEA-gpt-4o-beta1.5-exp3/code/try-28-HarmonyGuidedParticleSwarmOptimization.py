import numpy as np

class HarmonyGuidedParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, dim))
        self.local_best_positions = np.copy(self.positions)
        self.local_best_scores = np.full(self.swarm_size, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.func_evaluations = 0
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.harmony_memory_consideration_rate = 0.95

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                # Evaluate current position
                score = func(self.positions[i])
                self.func_evaluations += 1

                # Update local and global bests
                if score < self.local_best_scores[i]:
                    self.local_best_scores[i] = score
                    self.local_best_positions[i] = self.positions[i]
                
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            for i in range(self.swarm_size):
                # Harmony search influence
                if np.random.rand() < self.harmony_memory_consideration_rate:
                    new_position = self.local_best_positions[i] + np.random.normal(0, 0.1, self.dim)
                else:
                    new_position = self.positions[i]
                
                # Update velocity using PSO equation
                cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (self.local_best_positions[i] - self.positions[i])
                social_component = self.social_coeff * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component

                # Update position
                self.positions[i] = np.clip(new_position + self.velocities[i], self.lower_bound, self.upper_bound)

        return self.global_best_position