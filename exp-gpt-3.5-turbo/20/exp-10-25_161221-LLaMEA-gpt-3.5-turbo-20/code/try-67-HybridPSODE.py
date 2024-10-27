import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=30, c1=2.0, c2=2.0, f=0.5, cr=0.9):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.c1 = c1
        self.c2 = c2
        self.f = f
        self.cr = cr

    def __call__(self, func):
        def generate_candidate(population, gbest, c1, c2):
            r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
            v_individual = self.f * (population - population) + c1 * r1 * (gbest - population) + c2 * r2 * (population - gbest)
            return np.clip(population + v_individual, -5.0, 5.0)

        def differential_evolution(population, f, cr):
            mutant = population + f * (population[np.random.choice(range(self.swarm_size), 3, replace=False)] - population)
            crossover = np.random.rand(self.dim) < cr
            trial = np.where(crossover, mutant, population)
            return np.clip(trial, -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        gbest_idx = np.argmin(fitness)
        gbest = population[gbest_idx]

        for _ in range(self.budget):
            for idx in range(self.swarm_size):
                candidate = generate_candidate(population[idx], gbest, self.c1, self.c2)
                candidate_fitness = func(candidate)
                if candidate_fitness < fitness[idx]:
                    population[idx] = candidate
                    fitness[idx] = candidate_fitness
                    if candidate_fitness < fitness[gbest_idx]:
                        gbest = candidate
                        gbest_idx = idx
                else:
                    trial = differential_evolution(population[idx], self.f, self.cr)
                    trial_fitness = func(trial)
                    if trial_fitness < fitness[idx]:
                        population[idx] = trial
                        fitness[idx] = trial_fitness
                        if trial_fitness < fitness[gbest_idx]:
                            gbest = trial
                            gbest_idx = idx

        return gbest