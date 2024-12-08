import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Increased population for better exploration
        self.inertia_weight_initial = 0.9
        self.inertia_weight_final = 0.4
        self.cognitive_constant = 1.5
        self.social_constant = 1.5
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9  # Slightly increased to explore more diversity
        self.alpha = 1.5  # Levy flight parameter

    def levy_flight(self, size):
        sigma = (np.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                 (np.gamma((1 + self.alpha) / 2) * self.alpha *
                  2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
        u = np.random.normal(0, sigma, size)
        v = np.random.normal(0, 1, size)
        return u / abs(v) ** (1 / self.alpha)

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0

        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, particles)
            evaluations += self.population_size

            inertia_weight = (self.inertia_weight_final +
                              (self.inertia_weight_initial - self.inertia_weight_final) *
                              ((self.budget - evaluations) / self.budget))

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = particles[np.random.choice(idxs, 3, replace=False)]
                mutation_factor_dynamic = self.mutation_factor + np.random.rand() * 0.3
                mutant_vector = np.clip(a + mutation_factor_dynamic * (b - c), self.lower_bound, self.upper_bound)
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

            # Improved Local Search with Levy Flight
            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                levy_step = self.levy_flight(self.dim)
                local_candidate = particles[i] + levy_step
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

        return global_best_position, global_best_score