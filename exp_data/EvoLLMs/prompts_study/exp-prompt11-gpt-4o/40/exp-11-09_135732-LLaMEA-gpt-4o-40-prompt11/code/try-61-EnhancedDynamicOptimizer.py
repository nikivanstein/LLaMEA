import numpy as np

class EnhancedDynamicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.initial_population_size = 30  # Dynamic population starts smaller
        self.max_population_size = 60  # Allow population to grow
        self.inertia_weight = 0.5  # Slightly higher inertia for stability
        self.cognitive_constant = 1.3  # Adjusted cognitive constant
        self.social_constant = 1.5  # Adjusted social constant
        self.mutation_factor = 0.85  # Fine-tuned mutation factor
        self.crossover_rate = 0.85  # Adjusted crossover rate
        self.expansion_rate = 0.1  # Rate of increasing population size
        self.adaptive_switch_rate = 0.3  # Slightly increased adaptive switch rate

    def __call__(self, func):
        np.random.seed(42)
        population_size = self.initial_population_size
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        velocities = np.random.uniform(-0.5, 0.5, (population_size, self.dim))  # Adjusted velocity range
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        adaptive_phase = True

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

            r1 = np.random.rand(population_size, self.dim)
            r2 = np.random.rand(population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            if np.random.rand() < self.adaptive_switch_rate:
                adaptive_phase = not adaptive_phase

            if adaptive_phase:
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

            # Dynamic Local Search Phase
            for i in range(int(population_size * 0.2)):  # Reduced local search frequency
                if evaluations + 1 >= self.budget:
                    break
                local_candidate = particles[i] + np.random.uniform(-0.3, 0.3, self.dim)  # Broader local search range
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

            # Dynamically adjust population size
            if np.random.rand() < self.expansion_rate:
                new_particles = np.random.uniform(self.lower_bound, self.upper_bound, (5, self.dim))
                particles = np.vstack((particles, new_particles))
                velocities = np.vstack((velocities, np.random.uniform(-0.5, 0.5, (5, self.dim))))
                personal_best_positions = np.vstack((personal_best_positions, new_particles))
                personal_best_scores = np.append(personal_best_scores, np.full(5, np.inf))
                population_size = min(population_size + 5, self.max_population_size)

        return global_best_position, global_best_score