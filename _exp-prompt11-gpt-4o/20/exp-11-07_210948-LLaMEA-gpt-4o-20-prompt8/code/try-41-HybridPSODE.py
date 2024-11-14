import numpy as np

class HybridPSODE:
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
        # Combine positions and velocities initialization
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        scores = np.apply_along_axis(func, 1, positions)
        personal_best_scores = np.copy(scores)

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
            positions += velocities
            np.clip(positions, self.lower_bound, self.upper_bound, out=positions)

            new_scores = np.apply_along_axis(func, 1, positions)
            evaluations += self.population_size

            # Update personal and global bests
            mask = new_scores < personal_best_scores
            personal_best_scores[mask] = new_scores[mask]
            personal_best_positions[mask] = positions[mask]
            current_global_best_idx = np.argmin(personal_best_scores)
            if personal_best_scores[current_global_best_idx] < global_best_score:
                global_best_score = personal_best_scores[current_global_best_idx]
                global_best_position = personal_best_positions[current_global_best_idx]

            # DE Mutation and Crossover
            indices = np.random.choice(self.population_size, (self.population_size, 3), replace=True)
            a, b, c = indices.T
            mutant_vectors = np.clip(positions[a] + self.F * (positions[b] - positions[c]), self.lower_bound, self.upper_bound)
            crossover = np.random.rand(self.population_size, self.dim) < self.CR
            trials = np.where(crossover, mutant_vectors, positions)

            trial_scores = np.apply_along_axis(func, 1, trials)
            evaluations += self.population_size

            # Update population based on trial scores
            replace_mask = trial_scores < scores
            scores[replace_mask] = trial_scores[replace_mask]
            positions[replace_mask] = trials[replace_mask]

            if evaluations >= self.budget:
                break

        return global_best_position, global_best_score