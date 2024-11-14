import numpy as np

class HybridEvoSwarmOptimizer:
    def __init__(self, budget, dim, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lb = -5.0
        self.ub = 5.0
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))  # Initialized with small random velocities
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.inertia_weight = 0.9  # Start with a higher inertia weight
        self.cognitive_weight = 1.4
        self.social_weight = 1.6

    def __call__(self, func):
        evaluations = 0
        dynamic_inertia_weight = self.inertia_weight

        while evaluations < self.budget:
            for i in range(self.pop_size):
                fitness = func(self.positions[i])
                evaluations += 1
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]
                if evaluations >= self.budget:
                    break

            for i in range(self.pop_size):
                dynamic_inertia_weight = 0.5 + 0.4 * (self.budget - evaluations) / self.budget  # Adaptive inertia weight
                self.velocities[i] = (
                    dynamic_inertia_weight * self.velocities[i]
                    + self.cognitive_weight * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
                    + self.social_weight * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            for i in range(self.pop_size):
                if np.random.rand() < 0.15:  # Increased mutation probability
                    mutation_idx = np.random.randint(0, self.dim)
                    mutation_step = np.random.normal(0, 0.2)  # Larger mutation for more exploration
                    self.positions[i][mutation_idx] += mutation_step
                    self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

        return self.global_best_position, self.global_best_score