import numpy as np

class ImprovedDynamicHybridPSODEOptimizer:
    def __init__(self, budget, dim, mutation_factor=0.5, crossover_prob=0.7, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.mutation_factor, self.crossover_prob, self.w, self.c1, self.c2 = budget, dim, mutation_factor, crossover_prob, w, c1, c2

    def __call__(self, func):
        def fitness(x):
            return func(x)

        def create_population(swarm_size):
            return np.random.uniform(-5.0, 5.0, size=(swarm_size, self.dim))

        def PSO_DE_optimization():
            swarm_size = 30
            population = create_population(swarm_size)
            personal_best = population.copy()
            global_best = population[np.argmin([fitness(x) for x in population])]

            for _ in range(self.budget):
                w, c1, c2 = np.random.uniform(0, self.w), np.random.uniform(0, self.c1), np.random.uniform(0, self.c2)
                for i in range(swarm_size):
                    r1, r2 = np.random.rand(2, self.dim)
                    velocity = w * population[i] + c1 * r1 * (personal_best[i] - population[i]) + c2 * r2 * (global_best - population[i])
                    candidate = population[i] + velocity

                    candidate_fit = fitness(candidate)
                    personal_fit = fitness(personal_best[i])
                    global_fit = fitness(global_best)

                    if candidate_fit < personal_fit:
                        personal_best[i] = candidate

                    if candidate_fit < global_fit:
                        global_best = candidate

                    if np.random.rand() < self.crossover_prob:
                        mutant = population[np.random.choice(swarm_size, 3, replace=False)]
                        trial = population[i] + self.mutation_factor * (mutant[0] - mutant[1]) + self.mutation_factor * (mutant[2] - population[i])
                        trial_fit = fitness(trial)
                        if trial_fit < fitness(population[i]):
                            population[i] = trial

                swarm_size = int(30 + 20 * np.exp(-0.01 * _))  # Dynamic resizing

                if swarm_size > population.shape[0]:
                    population = np.vstack((population, create_population(swarm_size - population.shape[0])))
                elif swarm_size < population.shape[0]:
                    population = population[:swarm_size]

            return global_best

        return PSO_DE_optimization()