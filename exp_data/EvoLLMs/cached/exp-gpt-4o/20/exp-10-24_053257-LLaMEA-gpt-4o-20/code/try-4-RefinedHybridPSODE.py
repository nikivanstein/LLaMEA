import numpy as np

class RefinedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(50, budget // (dim * 5))
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.F = 0.5
        self.CR = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate current population
            for i, solution in enumerate(self.population):
                if self.evaluations >= self.budget:
                    break
                score = func(solution)
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = solution
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = solution

            # Update velocities and positions (PSO) with adaptive inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * (self.evaluations / self.budget))
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.population[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.population[i])
                self.velocities[i] = w * self.velocities[i] + cognitive_component + social_component
                self.population[i] = np.clip(self.population[i] + self.velocities[i], self.lower_bound, self.upper_bound)

            # Apply Differential Evolution mutation
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                indices = np.delete(np.arange(self.population_size), i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.where(np.random.rand(self.dim) < self.CR, mutant_vector, self.population[i])
                trial_score = func(trial_vector)
                self.evaluations += 1
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector

        return self.global_best_position, self.global_best_score