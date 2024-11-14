import numpy as np

class EnhancedDynamicHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Increased population size for improved exploration
        self.inertia_weight = 0.7  # Increased inertia for better global exploration
        self.cognitive_constant = 2.0  # Enhanced cognitive constant for quicker personal best convergence
        self.social_constant = 1.3  # Reduced social constant to balance exploration
        self.mutation_factor = 0.9  # Enhanced mutation factor for more aggressive exploration
        self.crossover_rate = 0.9  # Higher crossover rate for diversified solutions
        self.adaptive_switch_rate = 0.25  # Adjusted rate for adaptive strategy

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf
        evaluations = 0
        adaptive_phase = True
        dynamic_population_size = self.population_size

        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, particles[:dynamic_population_size])
            evaluations += dynamic_population_size

            for i in range(dynamic_population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            r1 = np.random.rand(dynamic_population_size, self.dim)
            r2 = np.random.rand(dynamic_population_size, self.dim)
            velocities[:dynamic_population_size] = (self.inertia_weight * velocities[:dynamic_population_size] +
                                                     self.cognitive_constant * r1 * (personal_best_positions[:dynamic_population_size] - particles[:dynamic_population_size]) +
                                                     self.social_constant * r2 * (global_best_position - particles[:dynamic_population_size]))
            particles[:dynamic_population_size] = np.clip(particles[:dynamic_population_size] + velocities[:dynamic_population_size], self.lower_bound, self.upper_bound)

            if np.random.rand() < self.adaptive_switch_rate:
                adaptive_phase = not adaptive_phase

            if adaptive_phase:
                for i in range(dynamic_population_size):
                    if evaluations + 1 >= self.budget:
                        break
                    idxs = [idx for idx in range(dynamic_population_size) if idx != i]
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

            # Dynamic Local Search Phase
            for i in range(dynamic_population_size // 3):  # Focused local search for convergence
                if evaluations + 1 >= self.budget:
                    break
                local_candidate = particles[i] + np.random.uniform(-0.2, 0.2, self.dim)
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

            # Dynamic population size adjustment
            if evaluations + dynamic_population_size < self.budget:
                dynamic_population_size = int(self.population_size * (1 - evaluations / self.budget))

        return global_best_position, global_best_score