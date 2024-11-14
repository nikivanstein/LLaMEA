import numpy as np

class EnhancedHybridPSO_SADE_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60  # Increased population for diversity
        self.inertia_weight = 0.7  # Adjusted inertia for better exploration/exploitation
        self.cognitive_constant = 1.4  # Slightly reduced for balance
        self.social_constant = 1.6  # Increased for better global search
        self.mutation_factor = 0.6  # Enhanced mutation for exploration
        self.crossover_rate = 0.9  # Higher crossover for more trial vectors

    def levy_flight(self, L):
        # Lévy flight step
        u = np.random.normal(0, 1, size=self.dim) * (0.01 / np.power(np.random.normal(0, 1), 1 / L))
        return u

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
            velocities = (self.inertia_weight * velocities +
                          self.cognitive_constant * r1 * (personal_best_positions - particles) +
                          self.social_constant * r2 * (global_best_position - particles))
            particles += velocities  # Removed clipping for free exploration

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

            # Lévy Flight Phase
            for i in range(self.population_size):
                if evaluations + 1 >= self.budget:
                    break
                step = self.levy_flight(1.5)
                levy_candidate = particles[i] + step
                levy_candidate = np.clip(levy_candidate, self.lower_bound, self.upper_bound)
                levy_score = func(levy_candidate)
                evaluations += 1
                if levy_score < scores[i]:
                    particles[i] = levy_candidate
                    scores[i] = levy_score

        return global_best_position, global_best_score