import numpy as np

class AdaptiveSwarmMutationOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 40
        self.c1 = 1.49445
        self.c2 = 1.49445
        self.w = 0.729
        self.F = 0.8
        self.CR = 0.9
        self.diversity_threshold = 0.1

    def __call__(self, func):
        np.random.seed(42)
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        velocities = np.random.uniform(-abs(self.upper_bound - self.lower_bound), abs(self.upper_bound - self.lower_bound), (self.population_size, self.dim))
        p_best_positions = np.copy(positions)
        p_best_scores = np.full(self.population_size, np.inf)
        g_best_position = None
        g_best_score = np.inf
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate the current population
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                score = func(positions[i])
                evaluations += 1
                if score < p_best_scores[i]:
                    p_best_scores[i] = score
                    p_best_positions[i] = positions[i]
                if score < g_best_score:
                    g_best_score = score
                    g_best_position = positions[i]

            # Calculate population diversity
            diversity = np.mean(np.std(positions, axis=0))

            # Update velocities and positions using PSO
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (self.w * velocities 
                          + self.c1 * r1 * (p_best_positions - positions) 
                          + self.c2 * r2 * (g_best_position - positions))
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            # Adaptive mutation strategy based on diversity
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break
                if diversity < self.diversity_threshold:
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)
                    mutant_vector = positions[a] + self.F * (positions[b] - positions[c])
                else:
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant_vector = positions[a] + self.F * (positions[b] - positions[c])

                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                trial_vector = np.copy(positions[i])
                for j in range(self.dim):
                    if np.random.rand() < self.CR:
                        trial_vector[j] = mutant_vector[j]
                trial_score = func(trial_vector)
                evaluations += 1
                if trial_score < p_best_scores[i]:
                    p_best_scores[i] = trial_score
                    positions[i] = trial_vector
                    if trial_score < g_best_score:
                        g_best_score = trial_score
                        g_best_position = trial_vector

        return g_best_position, g_best_score