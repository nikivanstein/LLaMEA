import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.initial_population_size = 32  # Slightly increased initial population
        self.final_population_size = 24  # More dynamic population resizing
        self.inertia_weight = 0.59  # Further reduced inertia weight for quicker convergence
        self.cognitive_coeff = 1.8  # Increased to further enhance exploration
        self.social_coeff = 1.3  # Slightly decreased to improve balance
        self.mutation_factor_init = 0.88  # Adjusted mutation factor for diversity
        self.cross_prob = 0.96  # Higher cross probability for more aggressive trial acceptance
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.initial_population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.initial_population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.initial_population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        eval_count = 0
        inertia_dampening = 0.99  # Adjusted for slightly faster dampening
        mutation_factor_change = (self.mutation_factor_init - 0.5) / self.budget

        while eval_count < self.budget:
            population_size = self.initial_population_size - int(
                (eval_count / self.budget) * (self.initial_population_size - self.final_population_size))

            for i in range(population_size):
                score = func(self.population[i])
                eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]
                if eval_count >= self.budget:
                    break

            r1 = np.random.rand(population_size, self.dim)
            r2 = np.random.rand(population_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions[:population_size] - self.population[:population_size])
            social_component = self.social_coeff * r2 * (self.global_best_position - self.population[:population_size])
            self.velocities[:population_size] = (self.inertia_weight * self.velocities[:population_size]) + cognitive_component + social_component
            self.population[:population_size] = self.population[:population_size] + self.velocities[:population_size]

            self.population[:population_size] = np.clip(self.population[:population_size], self.lower_bound, self.upper_bound)

            for i in range(population_size):
                indices = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                a, b, c = self.population[indices]
                mutant = a + (self.mutation_factor_init - mutation_factor_change * eval_count) * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])

                for j in range(self.dim):
                    if np.random.rand() < self.cross_prob:
                        trial[j] = mutant[j]

                trial_score = func(trial)
                eval_count += 1
                if trial_score < self.personal_best_scores[i]:
                    self.population[i] = trial
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial
                if eval_count >= self.budget:
                    break

            self.inertia_weight *= inertia_dampening

        return self.global_best_position, self.global_best_score