import numpy as np

class OptimizedAPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(max(20, dim * 5), 100)
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.4
        self.social_coeff = 1.4
        self.vel_clamp = (-(self.upper_bound - self.lower_bound), (self.upper_bound - self.lower_bound))
        self.eval_count = 0
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(self.vel_clamp[0], self.vel_clamp[1], (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility
        random_matrix1 = np.random.rand(self.budget, self.dim)
        random_matrix2 = np.random.rand(self.budget, self.dim)
        budget_index = 0

        while self.eval_count < self.budget:
            scores = np.apply_along_axis(func, 1, self.positions[self.eval_count:min(self.eval_count + self.population_size, self.budget)])
            for i, score in enumerate(scores):
                if self.eval_count >= self.budget:
                    break

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

                self.eval_count += 1

            for i in range(len(scores)):
                if self.eval_count >= self.budget:
                    break

                r1 = random_matrix1[budget_index % self.budget]
                r2 = random_matrix2[budget_index % self.budget]

                cognitive_velocity = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_coeff * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i]) + cognitive_velocity + social_velocity

                self.velocities[i] = np.clip(self.velocities[i], self.vel_clamp[0], self.vel_clamp[1])
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

                budget_index += 1

        return self.global_best_position, self.global_best_score