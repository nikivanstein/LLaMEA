import numpy as np

class ImprovedHybridPSODEOptimizer:
    def __init__(self, budget, dim, swarm_size=30, mutation_factor=0.5, crossover_prob=0.7, w=0.5, c1=1.5, c2=1.5):
        self.budget, self.dim, self.swarm_size, self.mutation_factor, self.crossover_prob, self.w, self.c1, self.c2 = budget, dim, swarm_size, mutation_factor, crossover_prob, w, c1, c2

    def __call__(self, func):
        def fitness(x):
            return func(x)

        def create_population():
            return np.random.uniform(-5.0, 5.0, size=(self.swarm_size, self.dim))

        population = create_population()
        personal_best = population.copy()
        global_best = population[np.argmin([fitness(x) for x in population])]

        for _ in range(self.budget):
            w, c1, c2 = np.random.uniform(0, self.w), np.random.uniform(0, self.c1), np.random.uniform(0, self.c2)
            for i in range(self.swarm_size):
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
                    mutant = population[np.random.choice(self.swarm_size, 3, replace=False)]
                    trial = population[i] + self.mutation_factor * (mutant[0] - mutant[1]) + self.mutation_factor * (mutant[2] - population[i])
                    trial_fit = fitness(trial)
                    if trial_fit < fitness(population[i]):
                        population[i] = trial

        return global_best