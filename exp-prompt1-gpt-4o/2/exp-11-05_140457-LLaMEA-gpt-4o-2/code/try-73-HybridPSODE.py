import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + 2 * int(np.sqrt(dim))
        self.min_bound = -5.0
        self.max_bound = 5.0
        self.F = 0.5  # DE scaling factor
        self.CR = 0.9  # crossover probability
        self.w = 0.5  # inertia weight for PSO
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.min_bound, self.max_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(x) for x in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evaluations = self.population_size

        while evaluations < self.budget:
            # PSO Update
            r1, r2 = np.random.rand(2)
            # Adaptive inertia weight based on diversity
            diversity = np.std(positions, axis=0).mean()
            adaptive_w = 0.5 + 0.5 * (diversity / (self.max_bound - self.min_bound))
            dynamic_c1 = 1.2 + 0.6 * np.random.random()  # Dynamic cognitive component
            dynamic_c2 = 1.2 + 0.6 * np.random.random()  # Dynamic social component
            velocities = (adaptive_w * velocities +
                          dynamic_c1 * r1 * (personal_best_positions - positions) +
                          dynamic_c2 * r2 * (global_best_position - positions))
            positions += velocities
            positions = np.clip(positions, self.min_bound, self.max_bound)

            # Evaluate PSO positions
            scores = np.array([func(x) for x in positions])
            evaluations += self.population_size
            improved = scores < personal_best_scores
            personal_best_positions[improved] = positions[improved]
            personal_best_scores[improved] = scores[improved]

            if np.min(scores) < global_best_score:
                global_best_position = positions[np.argmin(scores)]
                global_best_score = np.min(scores)

            # DE Update
            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = positions[np.random.choice(indices, 3, replace=False)]
                # Dynamic scaling factor based on diversity
                F_dynamic = 0.4 + 0.6 * (diversity / (self.max_bound - self.min_bound))
                mutant_vector = np.clip(a + F_dynamic * (b - c), self.min_bound, self.max_bound)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover_mask, mutant_vector, positions[i])
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < scores[i]:
                    positions[i] = trial_vector
                    scores[i] = trial_score
                    if trial_score < personal_best_scores[i]:
                        personal_best_scores[i] = trial_score
                        personal_best_positions[i] = trial_vector

                if trial_score < global_best_score:
                    global_best_position = trial_vector
                    global_best_score = trial_score

        return global_best_position, global_best_score