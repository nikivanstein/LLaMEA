import numpy as np

class ProbabilisticRefinedHybridDESA(RefinedHybridDESA):
    def __init__(self, budget, dim, mutation_prob=0.35):
        super().__init__(budget, dim, mutation_prob)

    def _mutation(self, population, target_index):
        candidates = [idx for idx in range(len(population)) if idx != target_index]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        # Probabilistic refinement
        if np.random.rand() < self.mutation_prob:
            mutation_offset = np.random.uniform(-0.1, 0.1, self.dim)
            mutant += mutation_offset
        return mutant

probabilistic_refined_hybrid_desa = ProbabilisticRefinedHybridDESA(budget=1000, dim=10)