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
        # Initialize particles and population
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        scores = np.full(self.population_size, np.inf)
        
        evaluations = 0
        while evaluations < self.budget:
            # Evaluate new positions using vectorized operations
            current_scores = np.apply_along_axis(func, 1, positions)
            evaluations += self.population_size
            
            # Update personal bests
            improved = current_scores < scores
            scores[improved] = current_scores[improved]
            personal_best_positions[improved] = positions[improved]
            
            # Update global best
            best_idx = np.argmin(scores)
            global_best_position = personal_best_positions[best_idx]
            
            r1, r2 = np.random.rand(2, self.population_size, self.dim)

            # PSO Update (combine position update and velocity calculation)
            velocities = (self.w * velocities +
                          self.c1 * r1 * (personal_best_positions - positions) +
                          self.c2 * r2 * (global_best_position - positions))
            positions += velocities
            np.clip(positions, self.lower_bound, self.upper_bound, out=positions)

            # DE Mutation and Crossover
            for i in range(self.population_size):
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                mutant = np.clip(personal_best_positions[indices[0]] + self.F * (personal_best_positions[indices[1]] - personal_best_positions[indices[2]]), self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, personal_best_positions[i])

                trial_score = func(trial)
                evaluations += 1
                if trial_score < scores[i]:
                    scores[i] = trial_score
                    personal_best_positions[i] = trial

            if evaluations >= self.budget:
                break

        return global_best_position, scores[best_idx]