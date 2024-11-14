import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.c1 = 1.4
        self.c2 = 1.4
        self.w = 0.6
        self.F = 0.5
        self.CR = 0.9
        self.velocities = np.zeros((self.population_size, self.dim))
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.scores = np.apply_along_axis(func, 1, self.positions)
        self.evaluations = self.population_size
        self.global_best_position = self.personal_best_positions[np.argmin(self.scores)]

    def __call__(self, func):
        while self.evaluations < self.budget:
            self._update_velocities()
            self._update_positions()
            self._evaluate_population(func)

            if self.evaluations >= self.budget:
                break

            self._differential_evolution(func)

        return self.global_best_position, self.scores[np.argmin(self.scores)]

    def _update_velocities(self):
        r1, r2 = np.random.rand(2, self.population_size, self.dim)
        self.velocities *= self.w
        self.velocities += self.c1 * r1 * (self.personal_best_positions - self.positions)
        self.velocities += self.c2 * r2 * (self.global_best_position - self.positions)

    def _update_positions(self):
        np.add(self.positions, self.velocities, out=self.positions)
        np.clip(self.positions, self.lower_bound, self.upper_bound, out=self.positions)

    def _evaluate_population(self, func):
        current_scores = np.apply_along_axis(func, 1, self.positions)
        self.evaluations += self.population_size
        improved = current_scores < self.scores
        self.scores[improved] = current_scores[improved]
        self.personal_best_positions[improved] = self.positions[improved]
        self.global_best_position = self.personal_best_positions[np.argmin(self.scores)]

    def _differential_evolution(self, func):
        for i in range(self.population_size):
            if self.evaluations >= self.budget:
                break
            idxs = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
            mutant = np.clip(self.personal_best_positions[idxs[0]] + self.F * (self.personal_best_positions[idxs[1]] - self.personal_best_positions[idxs[2]]), self.lower_bound, self.upper_bound)
            crossover_mask = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover_mask, mutant, self.personal_best_positions[i])
            trial_score = func(trial)
            self.evaluations += 1
            if trial_score < self.scores[i]:
                self.scores[i] = trial_score
                self.personal_best_positions[i] = trial