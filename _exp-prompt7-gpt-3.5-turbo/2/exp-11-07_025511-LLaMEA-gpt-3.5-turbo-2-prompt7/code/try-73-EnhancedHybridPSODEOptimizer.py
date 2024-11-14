import numpy as np

class EnhancedHybridPSODEOptimizer:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_prob=0.7, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.w, self.c1, self.c2 = budget, dim, swarm_size, mutation_factor, crossover_prob, w, c1, c2

    def __call__(self, func):
        def fitness(x):
            return func(x)

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
                population += velocity

                candidate_fit = fitness(population)
                personal_fit = fitness(personal_best)
                global_fit = fitness(global_best)

                personal_update_mask = candidate_fit < personal_fit
                personal_best[personal_update_mask] = population[personal_update_mask]

                global_update_mask = candidate_fit < global_fit
                global_best = np.where(global_update_mask.reshape(-1, 1).repeat(self.dim, axis=1), population, global_best)

                crossover_mask = np.random.rand(self.swarm_size) < self.crossover_prob
                crossover_idx = np.where(crossover_mask)[0]
                if len(crossover_idx) > 0:
                    mutants = np.random.choice(self.swarm_size, (len(crossover_idx), 3), replace=False)
                    trials = population[crossover_idx] + self.mutation_factor * (population[mutants[:, 0]] - population[mutants[:, 1]]) + self.mutation_factor * (population[mutants[:, 2]] - population[crossover_idx])
                    trial_fit = fitness(trials)
                    update_mask = trial_fit < fitness(population[crossover_idx])
                    population[crossover_idx] = np.where(update_mask.reshape(-1, 1).repeat(self.dim, axis=1), trials, population[crossover_idx])

            return global_best

        return PSO_DE_optimization()