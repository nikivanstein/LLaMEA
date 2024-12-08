import numpy as np

class AdaptiveQuantumHybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 60
        self.c1 = 1.49618
        self.c2 = 1.49618
        self.w_min = 0.4
        self.w_max = 0.9
        self.F = 0.8
        self.CR = 0.9
        self.rotational_diversity = 0.15
        self.mutation_probability = 0.2

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
            w = self.w_min + (self.w_max - self.w_min) * (np.sin(np.pi * evaluations / self.budget) ** 2)  # Chaotic inertia weight

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

            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            velocities = (w * velocities 
                          + self.c1 * r1 * (p_best_positions - positions) 
                          + self.c2 * r2 * (g_best_position - positions))
            positions = np.clip(positions + velocities, self.lower_bound, self.upper_bound)

            clone_count = int(self.rotational_diversity * self.population_size)
            for i in range(clone_count):
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant_vector = positions[a] + self.F * (positions[b] - positions[c])
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

            self.mutation_probability = 0.1 + 0.1 * np.cos(np.pi * evaluations / self.budget)  # Dynamic mutation probability
            if np.random.rand() < self.mutation_probability:
                mutation_indices = np.random.choice(self.population_size, size=int(0.2 * self.population_size), replace=False)
                for idx in mutation_indices:
                    mutation_vector = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                    positions[idx] = mutation_vector
                    score = func(positions[idx])
                    evaluations += 1
                    if score < p_best_scores[idx]:
                        p_best_scores[idx] = score
                        p_best_positions[idx] = positions[idx]
                        if score < g_best_score:
                            g_best_score = score
                            g_best_position = positions[idx]

        return g_best_position, g_best_score