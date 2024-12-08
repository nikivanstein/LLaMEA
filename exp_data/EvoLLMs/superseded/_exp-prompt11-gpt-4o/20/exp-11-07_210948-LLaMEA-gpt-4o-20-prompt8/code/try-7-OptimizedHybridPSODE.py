import numpy as np

class OptimizedHybridPSODE:
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
        # Initialize particles and DE population
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.full(self.population_size, np.inf)
        population = np.copy(positions)
        scores = np.copy(personal_best_scores)

        # Evaluate initial particles
        for i in range(self.population_size):
            score = func(positions[i])
            personal_best_scores[i] = score
            scores[i] = score

        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Combined PSO and DE Update
            for i in range(self.population_size):
                # PSO step
                if evaluations < self.budget:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = (self.w * velocities[i] +
                                     self.c1 * r1 * (personal_best_positions[i] - positions[i]) +
                                     self.c2 * r2 * (global_best_position - positions[i]))
                    positions[i] += velocities[i]
                    np.clip(positions[i], self.lower_bound, self.upper_bound, out=positions[i])
                    new_score = func(positions[i])
                    evaluations += 1
                    if new_score < personal_best_scores[i]:
                        personal_best_scores[i] = new_score
                        personal_best_positions[i] = positions[i]

                # DE step
                if evaluations < self.budget:
                    indices = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = indices[0], indices[1], indices[2]
                    mutant = population[a] + self.F * (population[b] - population[c])
                    np.clip(mutant, self.lower_bound, self.upper_bound, out=mutant)
                    trial = np.where(np.random.rand(self.dim) < self.CR, mutant, population[i])
                    trial_score = func(trial)
                    evaluations += 1
                    if trial_score < scores[i]:
                        scores[i] = trial_score
                        population[i] = trial

            # Update Global Best
            global_best_idx = np.argmin(personal_best_scores)
            global_best_position = personal_best_positions[global_best_idx]
            global_best_score = personal_best_scores[global_best_idx]

        return global_best_position, global_best_score