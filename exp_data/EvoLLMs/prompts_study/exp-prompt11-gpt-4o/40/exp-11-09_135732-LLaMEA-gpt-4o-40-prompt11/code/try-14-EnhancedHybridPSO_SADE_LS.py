import numpy as np

class EnhancedHybridPSO_SADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Increased population for better exploration
        self.inertia_weight = 0.7  # Adjusted for better exploration
        self.cognitive_constant = 1.4
        self.social_constant = 1.6  # Slightly increased to enhance convergence
        self.mutation_factor = 0.6  # Adjusted to enhance diversity
        self.crossover_rate = 0.9  # Increased for better exploration
        self.learning_rate = 0.1  # Q-learning rate for adaptive inertia weight

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

            for i in range(self.population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (
                self.inertia_weight * velocities +
                self.cognitive_constant * r1 * (personal_best_positions - particles) +
                self.social_constant * r2 * (global_best_position - particles)
            )
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

            # Enhanced Local Search with strategic step size adaptation
            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                step_size = np.random.uniform(-0.2, 0.2, self.dim)  # Increased step size range
                local_candidate = particles[i] + step_size
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

            # Adaptive inertia weight adjustment (Q-learning inspired)
            reward = np.mean(personal_best_scores) - global_best_score
            self.inertia_weight += self.learning_rate * reward
            self.inertia_weight = np.clip(self.inertia_weight, 0.4, 0.9)

        return global_best_position, global_best_score