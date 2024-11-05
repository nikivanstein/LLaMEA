import numpy as np

class HybridPSO_ADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.num_particles = 10 + 2 * int(np.sqrt(self.dim))
        self.w_max = 0.9  # max inertia weight
        self.w_min = 0.4  # min inertia weight
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.mutation_rate = 0.1
        self.v_max = 0.2 * (self.ub - self.lb)  # max velocity

        self.positions = np.random.uniform(self.lb, self.ub, (self.num_particles, dim))
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.num_particles, float('inf'))
        self.global_best_position = None
        self.global_best_score = float('inf')

        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            self.w = self.w_min + (self.w_max - self.w_min) * ((self.budget - self.evaluations) / self.budget)
            self.mutation_rate = 0.1 + 0.4 * (self.evaluations / self.budget)
            for i in range(self.num_particles):
                if self.evaluations >= self.budget:
                    break
                score = func(self.positions[i])
                self.evaluations += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            for i in range(self.num_particles):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity

                self.velocities[i] = np.clip(self.velocities[i], -self.v_max, self.v_max)

                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

                if np.random.rand() < self.mutation_rate * 0.8:  # Adjusted dynamic mutation rate
                    mutant = self.positions[i] + np.random.uniform(-0.5, 0.5, self.dim) * (self.global_best_position - self.positions[i])
                    mutant_score = func(np.clip(mutant, self.lb, self.ub))
                    self.evaluations += 1

                    if mutant_score < score:
                        self.positions[i] = np.clip(mutant, self.lb, self.ub)
                        if mutant_score < self.personal_best_scores[i]:
                            self.personal_best_scores[i] = mutant_score
                            self.personal_best_positions[i] = mutant.copy()

                        if mutant_score < self.global_best_score:
                            self.global_best_score = mutant_score
                            self.global_best_position = mutant.copy()

                adaptive_probability = 0.2 + 0.6 * (self.global_best_score / (self.personal_best_scores[i] + 1e-9))
                if np.random.rand() < adaptive_probability:  # Adaptive learning probability
                    local_search = self.positions[i] + np.random.uniform(-0.1, 0.1, self.dim)  # Local search step
                    local_search_score = func(np.clip(local_search, self.lb, self.ub))
                    self.evaluations += 1

                    if local_search_score < self.personal_best_scores[i]:
                        self.personal_best_scores[i] = local_search_score
                        self.personal_best_positions[i] = np.clip(local_search, self.lb, self.ub).copy()

        return self.global_best_position, self.global_best_score