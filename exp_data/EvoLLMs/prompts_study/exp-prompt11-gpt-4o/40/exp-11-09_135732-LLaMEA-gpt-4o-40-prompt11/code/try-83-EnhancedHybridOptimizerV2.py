import numpy as np

class EnhancedHybridOptimizerV2:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 50  # Initial larger population
        self.min_population_size = 20  # Minimum population size for adaptation
        self.inertia_weight = 0.5  # Slightly increased inertia weight
        self.cognitive_constant = 1.2  # Lower cognitive constant for balanced exploration
        self.social_constant = 1.7  # Slightly increased social constant for convergence
        self.mutation_factor = 0.9  # Higher mutation factor for exploration
        self.crossover_rate = 0.8  # Reduced crossover rate for selective mixing
        self.adaptive_switch_rate = 0.3  # Increased adaptive switching rate

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.initial_population_size, self.dim))  # Adjusted velocity range
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.initial_population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        adaptive_phase = True
        current_population_size = self.initial_population_size

        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, particles)
            evaluations += current_population_size

            for i in range(current_population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            r1 = np.random.rand(current_population_size, self.dim)
            r2 = np.random.rand(current_population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            if np.random.rand() < self.adaptive_switch_rate:
                adaptive_phase = not adaptive_phase

            if adaptive_phase:
                for i in range(current_population_size):
                    if evaluations + 1 >= self.budget:
                        break
                    idxs = [idx for idx in range(current_population_size) if idx != i]
                    a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                    mutant_vector = np.clip(a + self.mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                    trial_vector = np.copy(particles[i])
                    for j in range(self.dim):
                        if np.random.rand() < self.crossover_rate:
                            trial_vector[j] = mutant_vector[j]

                    trial_score = func(trial_vector)
                    evaluations += 1
                    if trial_score < scores[i]:
                        particles[i] = trial_vector
                        scores[i] = trial_score

            if evaluations + 1 >= self.budget:
                break

            for i in range(current_population_size // 3):
                if evaluations + 1 >= self.budget:
                    break
                local_candidate = particles[i] + np.random.uniform(-0.2, 0.2, self.dim)  # More refined local search range
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

            # Adaptive Population Reduction
            if current_population_size > self.min_population_size and np.random.rand() < 0.1:
                current_population_size -= 1
                particles = particles[:current_population_size]
                velocities = velocities[:current_population_size]
                personal_best_positions = personal_best_positions[:current_population_size]
                personal_best_scores = personal_best_scores[:current_population_size]

        return global_best_position, global_best_score