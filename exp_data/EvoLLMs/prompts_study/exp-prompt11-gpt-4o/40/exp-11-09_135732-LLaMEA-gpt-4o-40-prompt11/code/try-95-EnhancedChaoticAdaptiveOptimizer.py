import numpy as np

class EnhancedChaoticAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 40  # Reduced initial population size for faster convergence
        self.inertia_weight = 0.5  # Increased inertia weight for balanced exploration-exploitation
        self.cognitive_constant = 2.0  # Further increased cognitive constant for enhanced personal learning
        self.social_constant = 1.5  # Reduced social constant to balance global influence
        self.mutation_factor = 0.8  # Slightly reduced mutation factor for focused exploration
        self.crossover_rate = 0.9  # Increased crossover rate for more aggressive trials
        self.elitism_rate = 0.05  # Reduced elitism rate to allow wider exploration initially
    
    def logistic_map(self, z):
        return 4.0 * z * (1 - z)  # Logistic map for chaotic exploration

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (population_size, self.dim))  # Adjusted velocity range
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        z = np.random.rand()

        while evaluations < self.budget:
            scores = np.apply_along_axis(func, 1, particles)
            evaluations += population_size

            for i in range(population_size):
                if scores[i] < personal_best_scores[i]:
                    personal_best_scores[i] = scores[i]
                    personal_best_positions[i] = particles[i]
                if scores[i] < global_best_score:
                    global_best_score = scores[i]
                    global_best_position = particles[i]

            sorted_indices = np.argsort(scores)
            elite_count = int(self.elitism_rate * population_size)
            elite_indices = sorted_indices[:elite_count]

            r1 = np.random.rand(population_size, self.dim)
            r2 = np.random.rand(population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            for i in range(population_size):
                if evaluations + 1 >= self.budget:
                    break
                idxs = [idx for idx in range(population_size) if idx != i]
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

            for i in range(elite_count):  # Local search enhanced by chaotic map
                if evaluations + 1 >= self.budget:
                    break
                z = self.logistic_map(z)
                local_candidate = particles[elite_indices[i]] + np.random.uniform(-0.2, 0.2, self.dim) * z
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[elite_indices[i]]:
                    particles[elite_indices[i]] = local_candidate
                    scores[elite_indices[i]] = local_score

            # Adaptive increase in population size
            if evaluations < self.budget * 0.5:
                population_size = min(100, population_size + 5)

        return global_best_position, global_best_score