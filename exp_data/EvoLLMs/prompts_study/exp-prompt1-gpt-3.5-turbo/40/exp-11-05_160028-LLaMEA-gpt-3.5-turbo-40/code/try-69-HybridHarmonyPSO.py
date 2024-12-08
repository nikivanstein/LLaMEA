import numpy as np
import pyswarms as ps

class HybridHarmonyPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.budget, self.dim))
        fitness = np.array([func(x) for x in population])

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=self.budget, dimensions=self.dim, options=options, bounds=(self.lower_bound, self.upper_bound))

        for _ in range(self.budget - len(population)):
            new_harmony = optimizer.optimize(func, iters=1)[0]
            new_fitness = func(new_harmony)

            worst_idx = np.argmax(fitness)
            if new_fitness < fitness[worst_idx]:
                population[worst_idx] = new_harmony
                fitness[worst_idx] = new_fitness

        best_idx = np.argmin(fitness)
        return population[best_idx]