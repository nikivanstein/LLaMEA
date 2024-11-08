import numpy as np

class OptimizedPSODEOptimizer:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_prob=0.7, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.w, self.c1, self.c2 = budget, dim, swarm_size, mutation_factor, crossover_prob, w, c1, c2
        self.fit_cache = {}

    def __call__(self, func):
        def fitness(x):
            key = tuple(x)
            if key not in self.fit_cache:
                self.fit_cache[key] = func(x)
            return self.fit_cache[key]

        def create_population():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        def PSO_DE_optimization():
            population = create_population()
            personal_best = population.copy()
            global_best = population[np.argmin([fitness(x) for x in population])]

            for _ in range(self.budget):
                w, c1, c2 = np.random.uniform(0, self.w), np.random.uniform(0, self.c1), np.random.uniform(0, self.c2)
                r1, r2 = np.random.rand(2, self.swarm_size, self.dim)
                velocity = w * population + c1 * r1 * (personal_best - population) + c2 * r2 * (global_best - population)
                candidate = population + velocity

                candidate_fit = np.array([fitness(c) for c in candidate])
                personal_fit = np.array([fitness(p) for p in personal_best])
                global_fit = fitness(global_best)

                personal_mask = candidate_fit < personal_fit
                global_mask = candidate_fit < global_fit

                personal_best[personal_mask] = candidate[personal_mask]
                global_best = np.where(global_mask, candidate[global_mask], global_best)

                crossover_mask = np.random.rand(self.swarm_size) < self.crossover_prob
                mutants = population[np.random.choice(self.swarm_size, (self.swarm_size, 3), replace=False)]
                trials = population + self.mutation_factor * (mutants[:, 0] - mutants[:, 1]) + self.mutation_factor * (mutants[:, 2] - population)
                trial_fit = np.array([fitness(t) for t in trials])
                mask = trial_fit < np.array([fitness(p) for p in population])
                population[mask] = trials[mask]

            return global_best

        return PSO_DE_optimization()