import numpy as np
from scipy.stats import multivariate_normal

class EnhancedPSO_DE_CMAOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.velocity_clamp = (-2.0, 2.0)
        self.c1 = 1.7
        self.c2 = 1.3
        self.w = 0.4
        self.de_F = 0.7
        self.de_CR = 0.8
        self.adaptive_c1_rate = 0.04
        self.adaptive_c2_rate = 0.06
        self.adaptive_w_rate = 0.02
        self.adaptive_de_F_rate = 0.03
        self.num_clusters = 4
        self.restart_prob = 0.05
        self.neighborhood_size = 5
        self.cov_learning_rate = 0.05

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(self.velocity_clamp[0], self.velocity_clamp[1], (self.pop_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.pop_size, np.inf)
        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        cluster_assignments = np.random.choice(self.num_clusters, self.pop_size)
        cov_matrices = [np.eye(self.dim) for _ in range(self.num_clusters)]

        while evaluations < self.budget:
            scores = np.array([func(p) for p in particles])
            evaluations += self.pop_size

            for i in range(self.pop_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            self.c1 = max(0.3, self.c1 - self.adaptive_c1_rate * np.random.rand())
            self.c2 = min(2.3, self.c2 + self.adaptive_c2_rate * np.random.rand())
            self.w = max(0.2, self.w - self.adaptive_w_rate * np.random.rand())
            self.de_F = max(0.5, self.de_F + self.adaptive_de_F_rate * np.random.rand())

            if np.random.rand() < self.restart_prob:
                random_idx = np.random.choice(self.pop_size)
                particles[random_idx] = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                scores[random_idx] = func(particles[random_idx])
                evaluations += 1

            for s in range(self.num_clusters):
                cluster_indices = np.where(cluster_assignments == s)[0]
                if len(cluster_indices) > 1:
                    cluster_global_best_position = personal_best_positions[cluster_indices[np.argmin(personal_best_scores[cluster_indices])]]
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    for i in cluster_indices:
                        cognitive_component = self.c1 * r1 * (personal_best_positions[i] - particles[i])
                        social_component = self.c2 * r2 * (cluster_global_best_position - particles[i])
                        velocities[i] = np.clip(self.w * velocities[i] + cognitive_component + social_component,
                                                self.velocity_clamp[0], self.velocity_clamp[1])
                        particles[i] = np.clip(particles[i] + velocities[i], self.lower_bound, self.upper_bound)

                    cluster_mean = np.mean(particles[cluster_indices], axis=0)
                    cov_matrices[s] = (1 - self.cov_learning_rate) * cov_matrices[s] + self.cov_learning_rate * np.cov(particles[cluster_indices], rowvar=False)

            for i in range(self.pop_size):
                sorted_indices = np.argsort(personal_best_scores)
                neighborhood_indices = sorted_indices[max(0, i-self.neighborhood_size):min(self.pop_size, i+self.neighborhood_size)]
                a, b, c = np.random.choice(neighborhood_indices, 3, replace=False)
                mutant = np.clip(particles[a] + self.de_F * (particles[b] - particles[c]), self.lower_bound, self.upper_bound)
                trial = np.copy(particles[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.de_CR or j == j_rand:
                        trial[j] = mutant[j]
                trial_score = func(trial)
                if evaluations < self.budget and trial_score < scores[i]:
                    particles[i] = trial
                    scores[i] = trial_score
                    personal_best_positions[i] = trial
                    personal_best_scores[i] = trial_score
                    evaluations += 1

            for s in range(self.num_clusters):
                cluster_indices = np.where(cluster_assignments == s)[0]
                if len(cluster_indices) > 0:
                    new_positions = multivariate_normal.rvs(mean=particles[np.argmin(scores)], cov=cov_matrices[s], size=len(cluster_indices))
                    for i, pos in zip(cluster_indices, new_positions):
                        pos = np.clip(pos, self.lower_bound, self.upper_bound)
                        score = func(pos)
                        evaluations += 1
                        if evaluations < self.budget and score < scores[i]:
                            particles[i] = pos
                            scores[i] = score
                            personal_best_positions[i] = pos
                            personal_best_scores[i] = score

        return global_best_position, global_best_score