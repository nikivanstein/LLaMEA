import numpy as np

class EnhancedHybridPSODEOptimizer:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_prob=0.7, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.w, self.c1, self.c2 = budget, dim, swarm_size, mutation_factor, crossover_prob, w, c1, c2

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))
        personal_best = population.copy()
        global_best = population[np.argmin([func(x) for x in population])]

        for _ in range(self.budget):
            w, c1, c2 = np.random.uniform(0, self.w, self.swarm_size), np.random.uniform(0, self.c1, self.swarm_size), np.random.uniform(0, self.c2, self.swarm_size)
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            velocity = w[:, np.newaxis] * population + c1[:, np.newaxis] * r1 * (personal_best - population) + c2[:, np.newaxis] * r2 * (global_best - population)
            candidate = population + velocity

            candidate_fit = np.array([func(x) for x in candidate])
            personal_fit = np.array([func(x) for x in personal_best])
            global_fit = np.array([func(x) for x in [global_best]])

            personal_update = candidate_fit < personal_fit
            personal_best[personal_update] = candidate[personal_update]

            global_update = candidate_fit < global_fit
            global_best[global_update] = candidate[global_update]

            crossover_mask = np.random.rand(self.swarm_size) < self.crossover_prob
            mutant = population[np.random.choice(self.swarm_size, (3, self.swarm_size), replace=True)].T
            trial = population + self.mutation_factor * (mutant[:, 0] - mutant[:, 1])[:, np.newaxis] + self.mutation_factor * (mutant[:, 2] - population)
            trial_fit = np.array([func(x) for x in trial])
            trial_update = trial_fit < candidate_fit
            population[trial_update] = trial[trial_update]

        return global_best