import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 100
        self.inertia_weight = 0.5
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.mutation_factor = 0.8
        self.crossover_prob = 0.9
        self.vel_max = (self.upper_bound - self.lower_bound) * 0.1

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-self.vel_max, self.vel_max, (self.population_size, self.dim))
        personal_best_positions = np.copy(particles)
        personal_best_scores = np.full(self.population_size, float('inf'))
        
        for i in range(self.population_size):
            score = func(particles[i])
            personal_best_scores[i] = score

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]

        eval_count = self.population_size

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                velocities[i] = (self.inertia_weight * velocities[i]
                                 + self.cognitive_coef * r1 * (personal_best_positions[i] - particles[i])
                                 + self.social_coef * r2 * (global_best_position - particles[i]))
                velocities[i] = np.clip(velocities[i], -self.vel_max, self.vel_max)
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if eval_count >= self.budget:
                    break
                idxs = np.random.choice(self.population_size, 3, replace=False)
                while i in idxs:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = particles[idxs]
                adaptive_mutation = self.mutation_factor * (1 - eval_count / self.budget)
                mutant_vector = a + adaptive_mutation * (b - c)
                trial_vector = np.copy(particles[i])
                
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                trial_score = func(trial_vector)
                eval_count += 1

                if trial_score < personal_best_scores[i]:
                    personal_best_positions[i] = trial_vector
                    personal_best_scores[i] = trial_score

                if trial_score < personal_best_scores[global_best_index]:
                    global_best_position = trial_vector
                    global_best_index = i

        return global_best_position