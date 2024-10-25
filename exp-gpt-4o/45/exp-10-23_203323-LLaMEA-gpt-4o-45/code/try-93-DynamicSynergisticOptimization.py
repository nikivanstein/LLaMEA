import numpy as np

class DynamicSynergisticOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.num_particles = 60  # Changed number of particles
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.w_max = 0.8  # Adjusted inertia weight max
        self.w_min = 0.4  # Adjusted inertia weight min
        self.c1_initial = 2.1  # Altered cognitive component scaling
        self.c2_initial = 1.7
        self.c1_final = 1.4
        self.c2_final = 2.6  # Altered social component scaling
        self.velocity_clamp_initial = 1.0
        self.velocity_clamp_final = 0.3
        self.local_search_probability = 0.25  # Increased local search probability
        self.mutation_factor = 0.6  # Differentiated mutation factor
        self.crossover_rate = 0.85  # Slightly adjusted crossover rate

    def __call__(self, func):
        np.random.seed(42)  # Different seed for variability
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.num_particles, self.dim))
        velocities = np.random.uniform(-self.velocity_clamp_initial, self.velocity_clamp_initial, (self.num_particles, self.dim))
        personal_best_positions = positions.copy()
        personal_best_scores = np.full(self.num_particles, float('inf'))
        global_best_position = None
        global_best_score = float('inf')
        elite_position = None  # Track elite position

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
                        elite_position = global_best_position.copy()  # Update elite

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
                    trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector.copy()
                        if trial_score < global_best_score:
                            global_best_score = trial_score
                            global_best_position = trial_vector.copy()
                            elite_position = global_best_position.copy()  # Update elite

                cognitive_component = c1 * np.random.uniform(0, 1, self.dim) * (personal_best_positions[i] - positions[i])
                social_component = c2 * np.random.uniform(0, 1, self.dim) * (global_best_position - positions[i])
                elite_component = (0.1 * np.random.uniform(0, 1, self.dim) * (elite_position - positions[i])) if elite_position is not None else 0  # Elite component

                velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component + elite_component
                velocities[i] = np.clip(velocities[i], -velocity_clamp, velocity_clamp)
                positions[i] += velocities[i]
                positions[i] = np.clip(positions[i], self.lower_bound, self.upper_bound)