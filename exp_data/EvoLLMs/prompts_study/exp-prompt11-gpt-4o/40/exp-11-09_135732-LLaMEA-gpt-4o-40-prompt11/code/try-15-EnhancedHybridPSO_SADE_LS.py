import numpy as np

class EnhancedHybridPSO_SADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Increased population size
        self.inertia_weight_max = 0.9  # Dynamic inertia weight
        self.inertia_weight_min = 0.4
        self.cognitive_constant = 1.7  # Adjusted cognitive constant
        self.social_constant = 1.3  # Adjusted social constant
        self.mutation_factor = 0.6  # Enhanced mutation factor
        self.crossover_rate = 0.9  # Increased crossover rate

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

            inertia_weight = self.inertia_weight_max - ((self.inertia_weight_max - self.inertia_weight_min) * (evaluations / self.budget))
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
                crossover_indices = np.random.rand(self.dim) < self.crossover_rate
                trial_vector[crossover_indices] = mutant_vector[crossover_indices]

                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < scores[i]:
                    particles[i] = trial_vector
                    scores[i] = trial_score

            if evaluations + 1 >= self.budget:
                break

            # Optimized Local Search Phase
            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                local_candidate = particles[i] + np.random.uniform(-0.2, 0.2, self.dim)
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

        return global_best_position, global_best_score

# Example usage:
# optimizer = EnhancedHybridPSO_SADE_LS(budget=1000, dim=10)
# best_position, best_score = optimizer(some_black_box_function)