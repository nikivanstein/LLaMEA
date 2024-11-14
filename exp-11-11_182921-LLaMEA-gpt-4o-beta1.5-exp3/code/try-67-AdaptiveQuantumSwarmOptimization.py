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
        self.w = 0.5  # Inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.tau = 0.1  # Quantum perturbation probability

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                current_score = func(self.positions[i])
                self.func_evaluations += 1
                
                # Update personal best
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.positions[i]

            for i in range(self.population_size):
                # Update velocity
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (self.w * self.velocities[i] + cognitive_component + social_component)

                # Quantum perturbation
                if np.random.rand() < self.tau:
                    self.velocities[i] += np.random.normal(0, 1, self.dim)

                # Update position
                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            # Adaptive adjustment of inertia weight and tau
            self.w = 0.4 + 0.3 * np.sin(2 * np.pi * self.func_evaluations / self.budget)  # Adaptive inertia weight
            self.tau = 0.1 * (1 - np.cos(2 * np.pi * self.func_evaluations / self.budget))

        return self.global_best_position