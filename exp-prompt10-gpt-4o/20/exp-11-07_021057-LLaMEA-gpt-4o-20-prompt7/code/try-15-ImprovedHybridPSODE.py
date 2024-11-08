import numpy as np

class ImprovedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 80  # Reduced for efficiency
        self.inertia_weight = 0.7  # Adaptive inertia
        self.cognitive_coef = 2.0  # Increased for faster convergence
        self.social_coef = 2.0
        self.mutation_factor = 0.9  # Optimized mutation factor
        self.crossover_prob = 0.8

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros_like(particles)
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, float('inf'))
        
        for i in range(self.population_size):
            score = func(particles[i])
            personal_best_scores[i] = score

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        eval_count = self.population_size

        while eval_count < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities
                          + self.cognitive_coef * r1 * (personal_best_positions - particles)
                          + self.social_coef * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial_vector = np.where(crossover_mask, mutant_vector, particles[i])

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                    if trial_score < personal_best_scores[global_best_index]:
                        global_best_position = trial_vector
                        global_best_index = i

        return global_best_position