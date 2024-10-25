import numpy as np

class EnhancedAdaptiveHybridOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 60  # Increased number of particles
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.8  # Increased initial inertia weight
        self.w_min = 0.2  # Decreased minimum inertia weight
        self.c1_initial = 2.5  # Adjusted cognitive coefficient
        self.c2_initial = 1.5  # Adjusted social coefficient
        self.c1_final = 1.2  # Adjusted cognitive coefficient
        self.c2_final = 2.8  # Adjusted social coefficient
        self.velocity_clamp_initial = 0.95  # Increased initial velocity clamp
        self.velocity_clamp_final = 0.2  # Decreased final velocity clamp
        self.local_search_probability = 0.25  # Increased probability for local search
        self.mutation_factor = 0.6  # Adjusted differential mutation factor
        self.crossover_rate = 0.85  # Adjusted crossover rate for differential evolution

    def __call__(self, func):
        np.random.seed(0)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp_initial, self.velocity_clamp_initial, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')

        evaluations = 0
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

            for i in range(self.num_particles):
                if np.random.rand() < self.local_search_probability:
                    a, b, c = np.random.choice(self.num_particles, 3, replace=False)
                    mutant_vector = personal_best_positions[a] + self.mutation_factor * (personal_best_positions[b] - personal_best_positions[c])
                    trial_vector = np.where(np.random.rand(self.dim) < self.crossover_rate, mutant_vector, positions[i])
                    trial_vector = np.clip(trial_vector + np.random.normal(0, 0.1, self.dim), self.lower_bound, self.upper_bound)  # Added perturbation
                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector.copy()
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector.copy()

                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)
                
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)