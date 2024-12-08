import numpy as np

class EnhancedHybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.c1 = 1.5
        self.c2 = 1.5
        self.w = 0.7
        self.F = 0.5
        self.CR = 0.9

    def __call__(self, func):
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(pos) for pos in positions])
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            r1, r2 = np.random.rand(2, self.population_size, self.dim)

            # Vectorized PSO Update
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            # Evaluate new positions using vectorized approach
            new_scores = np.array([func(pos) for pos in positions])
            evaluations += self.population_size

            # Update personal bests
            better_scores_mask = new_scores < personal_best_scores
            personal_best_scores = np.where(better_scores_mask, new_scores, personal_best_scores)
            personal_best_positions = np.where(better_scores_mask[:, np.newaxis], positions, personal_best_positions)

            # DE Mutation and Crossover Vectorized
            indices = np.random.randint(0, self.population_size, (self.population_size, 3))
            a, b, c = indices[:, 0], indices[:, 1], indices[:, 2]
            mutants = np.clip(positions[a] + self.F * (positions[b] - positions[c]), self.lower_bound, self.upper_bound)
            crossover_mask = (np.random.rand(self.population_size, self.dim) < self.CR) | (np.arange(self.dim) == np.random.randint(self.dim, size=self.population_size)[:, None])
            trials = np.where(crossover_mask, mutants, positions)
            
            # Evaluate trials and update population
            trial_scores = np.array([func(trial) for trial in trials])
            evaluations += self.population_size
            scores = np.minimum(trial_scores, new_scores)
            positions = np.where(trial_scores[:, None] < new_scores[:, None], trials, positions)

            # Update Global Best
            current_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_best_idx]
                global_best_position = personal_best_positions[current_best_idx]

            if evaluations >= self.budget:
                break

        return global_best_position, global_best_score