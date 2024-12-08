import numpy as np

class AdaptiveMemeticParticleSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = int(10 + 2 * np.sqrt(dim))
        self.inertia_weight = 0.7
        self.cognitive_param = 1.5
        self.social_param = 1.5
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.array([float('inf')] * self.population_size)
        self.global_best_position = None
        self.global_best_score = float('inf')
        self.func_evaluations = 0

    def local_search(self, position, func):
        local_best_position = position
        local_best_score = func(position)
        step_size = 0.1
        for _ in range(3):  # Simple local search with few steps
            new_position = position + np.random.uniform(-step_size, step_size, self.dim)
            new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
            new_score = func(new_position)
            self.func_evaluations += 1
            if new_score < local_best_score:
                local_best_score = new_score
                local_best_position = new_position
        return local_best_position, local_best_score

    def __call__(self, func):
        while self.func_evaluations < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                self.velocities[i] = (
                    self.inertia_weight * self.velocities[i]
                    + self.cognitive_param * r1 * (self.personal_best_positions[i] - self.positions[i])
                    + self.social_param * r2 * (self.global_best_position - self.positions[i] if self.global_best_position is not None else 0)
                )
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

                current_score = func(self.positions[i])
                self.func_evaluations += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = np.copy(self.positions[i])
                    if current_score < self.global_best_score:
                        self.global_best_score = current_score
                        self.global_best_position = np.copy(self.positions[i])

            # Apply local search on global best to refine solution
            if self.global_best_position is not None:
                refined_position, refined_score = self.local_search(self.global_best_position, func)
                if refined_score < self.global_best_score:
                    self.global_best_score = refined_score
                    self.global_best_position = refined_position

            # Adaptive adjustment of inertia weight
            self.inertia_weight = 0.4 + 0.3 * (1 - self.func_evaluations / self.budget)

        return self.global_best_position