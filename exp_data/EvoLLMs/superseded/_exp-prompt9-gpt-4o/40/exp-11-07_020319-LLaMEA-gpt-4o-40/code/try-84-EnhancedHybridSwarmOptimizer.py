import numpy as np

class EnhancedHybridSwarmOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.initial_inertia = 0.9
        self.final_inertia = 0.4
        self.cognitive_coef = 1.6
        self.social_coef = 1.4
        self.mutation_factor_initial = 0.5
        self.mutation_factor_final = 1.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_position = positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)
        evaluations = self.population_size
        
        while evaluations < self.budget:
            progress = evaluations / self.budget
            inertia_weight = self.initial_inertia - progress * (self.initial_inertia - self.final_inertia)
            mutation_factor = self.mutation_factor_initial + progress * (self.mutation_factor_final - self.mutation_factor_initial)
            
            r1, r2 = np.random.rand(self.population_size, self.dim), np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities
                          + self.cognitive_coef * r1 * (personal_best_positions - positions)
                          + self.social_coef * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.lower_bound, self.upper_bound)

            scores = np.array([func(ind) for ind in positions])
            evaluations += self.population_size

            improved = scores < personal_best_scores
            personal_best_scores[improved] = scores[improved]
            personal_best_positions[improved] = positions[improved]

            current_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_best_idx]
                global_best_position = personal_best_positions[current_best_idx]

            if evaluations + self.population_size <= self.budget:
                for i in range(self.population_size):
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = np.random.choice(indices, 3, replace=False)
                    mutant_vector = np.clip(personal_best_positions[a] 
                                            + mutation_factor * (personal_best_positions[b] - personal_best_positions[c]),
                                            self.lower_bound, self.upper_bound)
                    crossover_mask = np.random.rand(self.dim) < 0.8
                    trial_vector = np.where(crossover_mask, mutant_vector, positions[i])

                    trial_score = func(trial_vector)
                    evaluations += 1

                    if trial_score < personal_best_scores[i]:
                        personal_best_positions[i] = trial_vector
                        personal_best_scores[i] = trial_score
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector

        return global_best_position, global_best_score