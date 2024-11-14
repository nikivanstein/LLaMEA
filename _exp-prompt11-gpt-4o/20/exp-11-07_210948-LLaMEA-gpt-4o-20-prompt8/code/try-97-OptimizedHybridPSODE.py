import numpy as np

class OptimizedHybridPSODE:
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

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros_like(positions)
        personal_best_positions = np.copy(positions)
        scores = np.apply_along_axis(func, 1, positions)
        evaluations = self.population_size

        global_best_position = personal_best_positions[np.argmin(scores)]

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)
            velocities = self.w * velocities + self.c1 * r1 * (personal_best_positions - positions) + self.c2 * r2 * (global_best_position - positions)
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            current_scores = np.apply_along_axis(func, 1, positions)
            evaluations += self.population_size

            better_mask = current_scores < scores
            scores[better_mask] = current_scores[better_mask]
            personal_best_positions[better_mask] = positions[better_mask]

            if evaluations >= self.budget:
                break

            global_best_position = personal_best_positions[np.argmin(scores)]

            r3 = np.random.rand(self.population_size, self.dim)
            mutation_candidates = np.random.choice(self.population_size, (self.population_size, 3), replace=False)
            mutant_vectors = np.clip(personal_best_positions[mutation_candidates[:, 0]] +
                                     self.F * (personal_best_positions[mutation_candidates[:, 1]] -
                                               personal_best_positions[mutation_candidates[:, 2]]),
                                     self.lower_bound, self.upper_bound)

            crossover_mask = r3 < self.CR
            trials = np.where(crossover_mask, mutant_vectors, personal_best_positions)

            for i in range(self.population_size):
                trial_score = func(trials[i])
                evaluations += 1
                if trial_score < scores[i]:
                    scores[i] = trial_score
                    personal_best_positions[i] = trials[i]

        return global_best_position, scores[np.argmin(scores)]