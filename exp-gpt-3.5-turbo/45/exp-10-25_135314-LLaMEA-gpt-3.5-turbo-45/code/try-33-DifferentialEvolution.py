import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR

    def mutate(self, population, target_idx):
        idxs = np.arange(self.budget)
        idxs = idxs[idxs != target_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        return population[a] + self.F * (population[b] - population[c])

    def crossover(self, target, mutant):
        j_rand = np.random.randint(self.dim)
        trial = np.where(np.random.rand(self.dim) < self.CR, mutant, target)
        trial[j_rand] = mutant[j_rand]
        return trial

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        for _ in range(self.budget):
            next_population = np.empty_like(population)
            for target_idx in range(self.budget):
                mutant = self.mutate(population, target_idx)
                trial = self.crossover(population[target_idx], mutant)
                if func(trial) < func(population[target_idx]):
                    next_population[target_idx] = trial
                else:
                    next_population[target_idx] = population[target_idx]
            population = next_population

        return population[np.argmin([func(individual) for individual in population])]