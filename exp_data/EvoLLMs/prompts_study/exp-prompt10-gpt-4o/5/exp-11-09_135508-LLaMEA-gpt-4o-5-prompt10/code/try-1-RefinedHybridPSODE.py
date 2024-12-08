import numpy as np

class RefinedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 30
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.mutation_factor = 0.9  # Increased mutation factor for diversity
        self.cross_prob = 0.9
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf

    def __call__(self, func):
        eval_count = 0
        inertia_dampening = 0.99  # New parameter for adaptive inertia

        while eval_count < self.budget:
            # Evaluate the population
            for i in range(self.population_size):
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

            # Update velocities and positions using PSO
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions - self.population)
            social_component = self.social_coeff * r2 * (self.global_best_position - self.population)
            self.velocities = (self.inertia_weight * self.velocities) + cognitive_component + social_component
            self.population = self.population + self.velocities

            # Boundary handling
            self.population = np.clip(self.population, self.lower_bound, self.upper_bound)

            # Apply Differential Evolution
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = a + self.mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)
                trial = np.copy(self.population[i])

                for j in range(self.dim):
                    if np.random.rand() < self.cross_prob:
                        trial[j] = mutant[j]

                # Evaluate the trial vector
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

            # Adapt inertia weight
            self.inertia_weight *= inertia_dampening

        return self.global_best_position, self.global_best_score