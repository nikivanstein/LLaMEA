import numpy as np

class AdaptiveMultiSwirlPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.swirl_count = 3  # Number of swirls for local search
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.best_positions = np.copy(self.positions)
        self.global_best_position = None
        self.func_evaluations = 0
        self.global_best_score = float('inf')
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                current_score = func(self.positions[i])
                self.func_evaluations += 1

                if current_score < func(self.best_positions[i]):
                    self.best_positions[i] = np.copy(self.positions[i])

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = np.copy(self.positions[i])

            inertia_weight = self.w_max - (self.w_max - self.w_min) * (self.func_evaluations / self.budget)

            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive_component + social_component

                # Multi-swirl strategy
                for _ in range(self.swirl_count):
                    swirl_vector = np.random.normal(0, 0.1, self.dim)
                    if np.random.rand() < 0.5:
                        self.velocities[i] += swirl_vector

                self.positions[i] = np.clip(self.positions[i] + self.velocities[i], self.lower_bound, self.upper_bound)

        return self.global_best_position