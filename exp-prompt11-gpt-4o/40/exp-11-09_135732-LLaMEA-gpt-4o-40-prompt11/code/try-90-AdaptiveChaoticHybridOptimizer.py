import numpy as np

class AdaptiveChaoticHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Slightly increased population size for better exploration
        self.initial_inertia_weight = 0.7  # Adaptive inertia weight starting value
        self.cognitive_constant = 2.0  # Enhanced cognitive learning rate
        self.social_constant = 1.5  # Reduced social constant for balance
        self.mutation_factor = 0.8  # Reduced mutation factor for focused diversity
        self.crossover_rate = 0.9  # Increased crossover rate to enhance recombination
        self.elitism_rate = 0.2  # Increased elitism rate

    def __call__(self, func):
        np.random.seed(42)
        # Chaotic initialization using logistic map
        chaotic_map = np.random.rand(self.population_size, self.dim)
        for i in range(100):  # Iterate to stabilize chaotic sequence
            chaotic_map = 4 * chaotic_map * (1 - chaotic_map)
        particles = self.lower_bound + (self.upper_bound - self.lower_bound) * chaotic_map
        velocities = np.random.uniform(-0.5, 0.5, (self.population_size, self.dim))
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        iterations = 0

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

            sorted_indices = np.argsort(scores)
            elite_count = int(self.elitism_rate * self.population_size)
            elite_indices = sorted_indices[:elite_count]

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            # Adapt inertia weight over iterations
            inertia_weight = self.initial_inertia_weight - (iterations / (self.budget / self.population_size)) * 0.4
            velocities = (inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
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

            for i in range(elite_count):  # Local search only on elite solutions
                if evaluations + 1 >= self.budget:
                    break
                local_candidate = particles[elite_indices[i]] + np.random.uniform(-0.1, 0.1, self.dim)
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[elite_indices[i]]:
                    particles[elite_indices[i]] = local_candidate
                    scores[elite_indices[i]] = local_score

            iterations += 1

        return global_best_position, global_best_score