import numpy as np

class AdaptiveSwarmBasedQuantumOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = int(10 + 2 * np.sqrt(dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, dim))
        self.func_evaluations = 0
        self.best_score = float('inf')
        self.best_position = np.random.uniform(self.lower_bound, self.upper_bound, dim)
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, float('inf'))
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.tau = 0.1  # Probability for quantum update

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.swarm_size):
                current_score = func(self.positions[i])
                self.func_evaluations += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]
                
                if current_score < self.best_score:
                    self.best_score = current_score
                    self.best_position = self.positions[i]

            for i in range(self.swarm_size):
                inertia_component = self.inertia_weight * self.velocities[i]
                cognitive_component = self.cognitive_param * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.social_param * np.random.rand(self.dim) * (self.best_position - self.positions[i])
                self.velocities[i] = inertia_component + cognitive_component + social_component
                
                # Quantum-inspired positional update
                if np.random.rand() < self.tau:
                    self.positions[i] += np.random.normal(0, 1, self.dim)
                else:
                    self.positions[i] += self.velocities[i]
                
                # Clamp positions to bounds
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Adaptive parameter adjustment
            self.inertia_weight = 0.4 + (0.3 * np.cos(2 * np.pi * self.func_evaluations / self.budget))
            self.tau = 0.05 + (0.05 * np.sin(2 * np.pi * self.func_evaluations / self.budget))
        
        return self.best_position