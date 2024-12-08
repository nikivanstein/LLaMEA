import numpy as np

class EnhancedAdaptiveSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 25  # Adjusted population size
        self.inertia_weight = 0.6  # Adjusted inertia weight
        self.c1 = 0.8  # Simplified adaptive coefficients
        self.c2 = 1.8
        self.c3 = 0.5  # New coefficient for random exploration
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))  # Initialized to zero
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_position = positions[np.argmin(personal_best_scores)]
        global_best_score = min(personal_best_scores)
        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2, r3 = np.random.rand(3, self.population_size, self.dim)
            random_exploration = r3 * self.c3 * (np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim)) - positions)
            velocities = (self.inertia_weight * velocities
                          + self.c1 * r1 * (personal_best_positions - positions)
                          + self.c2 * r2 * (global_best_position - positions)
                          + random_exploration)
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            scores = np.array([func(ind) for ind in positions])
            evaluations += self.population_size

            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]

            min_score_idx = np.argmin(personal_best_scores)
            if personal_best_scores[min_score_idx] < global_best_score:
                global_best_score = personal_best_scores[min_score_idx]
                global_best_position = personal_best_positions[min_score_idx]

        return global_best_position, global_best_score