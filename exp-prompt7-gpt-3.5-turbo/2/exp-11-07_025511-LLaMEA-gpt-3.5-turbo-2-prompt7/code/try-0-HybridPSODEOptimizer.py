import numpy as np

class HybridPSODEOptimizer:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_prob=0.7, w=0.5, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.w = w
        self.c1 = c1
        self.c2 = c2

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
                w = np.random.uniform(0, self.w)
                c1 = np.random.uniform(0, self.c1)
                c2 = np.random.uniform(0, self.c2)
                for i in range(self.swarm_size):
                    r1, r2 = np.random.uniform(0, 1, size=(2, self.dim))
                    velocity = w * population[i] + c1 * r1 * (personal_best[i] - population[i]) + c2 * r2 * (global_best - population[i])
                    candidate = population[i] + velocity

                    if fitness(candidate) < fitness(personal_best[i]):
                        personal_best[i] = candidate

                    if fitness(candidate) < fitness(global_best):
                        global_best = candidate

                    if np.random.uniform() < self.crossover_prob:
                        mutant = population[np.random.choice(self.swarm_size, 3, replace=False)]
                        trial = population[i] + self.mutation_factor * (mutant[0] - mutant[1]) + self.mutation_factor * (mutant[2] - population[i])
                        if fitness(trial) < fitness(population[i]):
                            population[i] = trial

            return global_best

        return PSO_DE_optimization()