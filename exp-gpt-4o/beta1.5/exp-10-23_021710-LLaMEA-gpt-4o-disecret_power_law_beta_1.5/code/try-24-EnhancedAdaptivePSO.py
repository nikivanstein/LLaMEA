import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.c1 = 1.5  # Adjusted cognitive coefficient
        self.c2 = 2.5  # Adjusted social coefficient
        self.inertia_weight = 0.9  # Dynamic inertia weight for better exploration initially
        self.inertia_dampening = 0.99
        self.max_evaluations = budget

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.max_evaluations:
            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1
                
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]
                
                if evaluations >= self.max_evaluations:
                    break

            for i in range(self.population_size):
                neighbors = np.random.choice(self.population_size, size=3, replace=False)
                local_best_position = self.personal_best_positions[neighbors[np.argmin([self.personal_best_scores[j] for j in neighbors])]]
                
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)

                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i])
                social_component = self.c2 * r2 * (local_best_position - self.particles[i])

                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component
                self.particles[i] += self.velocities[i]

                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            if evaluations < self.max_evaluations:
                for i in range(self.population_size):
                    if np.random.rand() < 0.2:  # Probability to explore
                        random_step = np.random.normal(0, 0.1, self.dim)  # Reduced variance for fine-tuning
                        candidate_vector = np.clip(self.particles[i] + random_step, self.lower_bound, self.upper_bound)
                        candidate_score = func(candidate_vector)
                        evaluations += 1

                        if candidate_score < self.personal_best_scores[i]:
                            self.particles[i] = candidate_vector
                            self.personal_best_scores[i] = candidate_score
                            self.personal_best_positions[i] = candidate_vector

                        if candidate_score < self.global_best_score:
                            self.global_best_score = candidate_score
                            self.global_best_position = candidate_vector

                        if evaluations >= self.max_evaluations:
                            break

            self.inertia_weight *= self.inertia_dampening
        
        return self.global_best_position, self.global_best_score