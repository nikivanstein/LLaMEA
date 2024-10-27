import numpy as np

class RefinedHybridDESA(HybridDESA):
    def _mutation(self, population, target_index):
        candidates = [idx for idx in range(len(population)) if idx != target_index]
        a, b, c, d = np.random.choice(candidates, 4, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c]) + self.mutation_factor * (population[d] - population[a])
        return mutant