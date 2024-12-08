import numpy as np

class ImprovedHybridPSODEOptimizer:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_prob=0.7, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.w, self.c1, self.c2 = budget, dim, swarm_size, mutation_factor, crossover_prob, w, c1, c2

    def __call__(self, func):
        def fitness(x):
            return func(x)

        population = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        personal_best = population.copy()
        global_best = population[np.argmin([fitness(x) for x in population])]

        for _ in range(self.budget):
            w, c1, c2 = np.random.uniform(0, self.w, self.swarm_size), np.random.uniform(0, self.c1, self.swarm_size), np.random.uniform(0, self.c2, self.swarm_size)
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            mutant_indices = np.random.choice(self.swarm_size, (self.swarm_size, 3), replace=False)

            velocity = w[:, None] * population + c1[:, None] * r1 * (personal_best - population) + c2[:, None] * r2 * (global_best - population)
            candidates = population + velocity

            candidate_fits = np.array([fitness(c) for c in candidates])
            personal_fits = np.array([fitness(p) for p in personal_best])
            global_fits = fitness(global_best)

            personal_update_mask = candidate_fits < personal_fits
            personal_best[personal_update_mask] = candidates[personal_update_mask]

            global_update_mask = candidate_fits < global_fits
            global_best = np.where(global_update_mask, candidates, global_best)

            crossover_mask = np.random.rand(self.swarm_size) < self.crossover_prob
            mutants = population[mutant_indices]
            trials = population + self.mutation_factor * (mutants[:, :, 0] - mutants[:, :, 1]) + self.mutation_factor * (mutants[:, :, 2] - population)
            trial_fits = np.array([fitness(t) for t in trials])
            mutation_mask = trial_fits < candidate_fits
            population[mutation_mask] = trials[mutation_mask]

        return global_best