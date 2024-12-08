import numpy as np

class AdaptiveDEAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 * dim
        self.F = 0.5
        self.CR = 0.9
        self.strategy_prob = 0.16666666666666666

    def mutate(self, target_idx, population):
        idxs = [idx for idx in range(len(population)) if idx != target_idx]
        a, b, c = population[np.random.choice(idxs, 3, replace=False)]
        mutant = np.clip(a + self.F * (b - c), -5, 5)
        return mutant

    def crossover(self, target, mutant):
        trial = np.copy(target)
        for i in range(self.dim):
            if np.random.rand() > self.CR:
                trial[i] = mutant[i]
        return trial

    def fitness(self, func, candidate):
        return func(candidate)

    def __call__(self, func):
        population = np.random.uniform(-5, 5, (self.pop_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.pop_size):
                target = population[i]
                mutant = self.mutate(i, population)
                trial = self.crossover(target, mutant)
                if self.fitness(func, trial) < self.fitness(func, target):
                    population[i] = trial
        return population[np.argmin([self.fitness(func, p) for p in population])]