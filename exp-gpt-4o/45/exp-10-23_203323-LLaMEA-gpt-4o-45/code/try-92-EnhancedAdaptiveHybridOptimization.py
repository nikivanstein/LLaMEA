import numpy as np

class EnhancedAdaptiveHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 50
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.8
        self.w_min = 0.2
        self.c1_initial = 2.5
        self.c2_initial = 1.5
        self.c1_final = 1.0
        self.c2_final = 2.8
        self.velocity_clamp_initial = 1.0
        self.velocity_clamp_final = 0.2
        self.local_search_probability = 0.25
        self.mutation_factor_initial = 0.6
        self.mutation_factor_final = 0.3
        self.crossover_rate_initial = 0.8
        self.crossover_rate_final = 0.95
        self.chaotic_map = lambda x: 0.9 * x * (1 - x)  # Logistic map

    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp_initial, self.velocity_clamp_initial, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
        chaotic_value = 0.5
        while evaluations < self.budget:
            for i in range(self.num_particles):
                score = func(positions[i])
                evaluations += 1
                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i].copy()
                    if score < global_best_score:
                        global_best_score = score
                        global_best_position = positions[i].copy()

                if evaluations >= self.budget:
                    break

            inertia_weight = self.w_max - ((self.w_max - self.w_min) * evaluations / self.budget)
            c1 = self.c1_initial - ((self.c1_initial - self.c1_final) * evaluations / self.budget)
            c2 = self.c2_initial + ((self.c2_final - self.c2_initial) * evaluations / self.budget)
            velocity_clamp = self.velocity_clamp_initial - ((self.velocity_clamp_initial - self.velocity_clamp_final) * evaluations / self.budget)
            mutation_factor = self.mutation_factor_initial - (self.mutation_factor_initial - self.mutation_factor_final) * evaluations / self.budget
            crossover_rate = self.crossover_rate_initial + (self.crossover_rate_final - self.crossover_rate_initial) * evaluations / self.budget

            chaotic_value = self.chaotic_map(chaotic_value)

            for i in range(self.num_particles):
                if np.random.rand() < self.local_search_probability:
                    a, b, c = np.random.choice(self.num_particles, 3, replace=False)
                    mutant_vector = personal_best_positions[a] + mutation_factor * (personal_best_positions[b] - personal_best_positions[c])
                    trial_vector = np.where(np.random.rand(self.dim) < crossover_rate, mutant_vector, positions[i])
                    trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector.copy()
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector.copy()

                cognitive_component = c1 * chaotic_value * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * chaotic_value * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)