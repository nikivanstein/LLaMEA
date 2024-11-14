import numpy as np

class HybridEvoSwarmOptimizer:
    def __init__(self, budget, dim, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.lb = -5.0
        self.ub = 5.0
        self.positions = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.velocities = np.zeros((self.pop_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.inertia_weight = 0.9  # Increased inertia weight for better exploration
        self.cognitive_weight = 1.5
        self.social_weight = 1.5
        self.min_inertia = 0.4  # Adaptive inertia weight bounds
        self.max_inertia = 0.9

    def __call__(self, func):
        evaluations = 0
        iter_count = 0

        while evaluations < self.budget:
            # Evaluate fitness for each particle
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

            # Adaptive weight based on iteration
            self.inertia_weight = self.max_inertia - ((self.max_inertia - self.min_inertia) * (evaluations / self.budget))
            
            # Update velocities and positions
            for i in range(self.pop_size):
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i]
                    + self.cognitive_weight * np.random.rand(self.dim) * (self.personal_best_positions[i] - self.positions[i])
                    + self.social_weight * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

            # Perform selective evolutionary mutation with adjusted probability
            for i in range(self.pop_size):
                if np.random.rand() < 0.05:  # Reduced mutation probability
                    mutation_idx = np.random.randint(0, self.dim)
                    mutation_step = np.random.normal(0, 0.05)  # Smaller mutation
                    self.positions[i][mutation_idx] += mutation_step
                    self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

        return self.global_best_position, self.global_best_score