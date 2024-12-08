import numpy as np

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50  # Increased population size for better exploration
        self.inertia_weight = 0.5  # Dynamic inertia weight
        self.cognitive_constant = 1.2  # Reduced cognitive constant
        self.social_constant = 1.8  # Increased social constant
        self.mutation_factor = 0.9  # Enhanced mutation factor
        self.crossover_rate = 0.85  # Slightly reduced crossover rate
        self.adaptive_switch_rate = 0.3  # Increased adaptive strategy switch rate

    def __call__(self, func):
        np.random.seed(42)
        particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-0.3, 0.3, (self.population_size, self.dim))  # Further reduced velocity range
        personal_best_positions = particles.copy()
        personal_best_scores = np.full(self.population_size, np.inf)

        global_best_position = None
        global_best_score = np.inf

        evaluations = 0
        adaptive_phase = True
        dynamic_scaling_factor = 0.5  # Initial dynamic scaling factor

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

            # Adjust inertia weight dynamically
            self.inertia_weight = 0.9 - (0.5 * evaluations / self.budget)
            
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles = np.clip(particles + velocities, self.lower_bound, self.upper_bound)

            if np.random.rand() < self.adaptive_switch_rate:
                adaptive_phase = not adaptive_phase

            if adaptive_phase:
                for i in range(self.population_size):
                    if evaluations >= self.budget:
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

            if evaluations >= self.budget:
                break

            # Enhanced Local Search Phase with dynamic scaling
            for i in range(self.population_size // 2):  # Increased local search participation
                if evaluations >= self.budget:
                    break
                local_candidate = particles[i] + np.random.uniform(-0.2, 0.2, self.dim) * dynamic_scaling_factor
                local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                local_score = func(local_candidate)
                evaluations += 1
                if local_score < scores[i]:
                    particles[i] = local_candidate
                    scores[i] = local_score

            dynamic_scaling_factor *= 0.95  # Decay the dynamic scaling factor for precision

        return global_best_position, global_best_score