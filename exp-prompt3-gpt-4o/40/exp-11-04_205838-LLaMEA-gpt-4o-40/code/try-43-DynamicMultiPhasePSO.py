import numpy as np

class DynamicMultiPhasePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = max(20, dim * 7)
        self.inertia_weight_max = 0.9
        self.inertia_weight_min = 0.3
        self.cognitive_coeff = 1.5  # Increased cognitive coefficient
        self.social_coeff = 1.5  # Balanced social coefficient
        self.de_mutation_prob = 0.2  # Probability for differential evolution mutation
        self.eval_count = 0

    def __call__(self, func):
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1.5, (self.population_size, self.dim))  # Adjusted velocity range

        personal_best_positions = particles.copy()
        personal_best_scores = np.array([func(p) for p in particles])
        self.eval_count += self.population_size

        global_best_index = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_index]
        global_best_score = personal_best_scores[global_best_index]

        phase_switch = self.budget // 2  # Switch phase halfway through the budget

        while self.eval_count < self.budget:
            if self.eval_count > phase_switch:
                r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
                velocities = (self.inertia_weight_max * velocities +
                              self.cognitive_coeff * r1 * (personal_best_positions - particles) +
                              self.social_coeff * r2 * (global_best_position - particles))
            else:
                velocities = (self.inertia_weight_min * velocities +
                              np.random.rand(self.population_size, self.dim) * (global_best_position - particles))

            particles += velocities
            particles = np.clip(particles, self.lower_bound, self.upper_bound)

            scores = np.array([func(p) for p in particles])
            self.eval_count += self.population_size

            better_mask = scores < personal_best_scores
            personal_best_positions[better_mask] = particles[better_mask]
            personal_best_scores[better_mask] = scores[better_mask]

            global_best_index = np.argmin(personal_best_scores)
            if personal_best_scores[global_best_index] < global_best_score:
                global_best_score = personal_best_scores[global_best_index]
                global_best_position = personal_best_positions[global_best_index]

            if np.random.rand() < self.de_mutation_prob:
                for i in range(self.population_size):
                    if self.eval_count >= self.budget:
                        break
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    mutant = personal_best_positions[indices[0]] + 0.5 * (personal_best_positions[indices[1]] - personal_best_positions[indices[2]])
                    mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                    mutant_score = func(mutant)
                    self.eval_count += 1
                    if mutant_score < personal_best_scores[i]:
                        personal_best_positions[i] = mutant
                        personal_best_scores[i] = mutant_score
                        if mutant_score < global_best_score:
                            global_best_score = mutant_score
                            global_best_position = mutant

        return global_best_position