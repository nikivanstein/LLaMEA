import numpy as np

class EnhancedOppositionBasedDE(EnhancedAdaptiveDE):
    def __init__(self, budget, dim):
        super().__init__(budget, dim)

    def initialize_population(self):
        self.pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.pop[1::2] = 5.0 - self.pop[1::2]  # Opposite solutions

    def mutate(self, pop, best, F):
        for i in range(self.pop_size):
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = pop[a] + F * (pop[b] - pop[c])
            for j in range(self.dim):
                if np.random.uniform() < self.CR or j == np.random.randint(0, self.dim):
                    pop[i, j] = mutant[j]
        return pop