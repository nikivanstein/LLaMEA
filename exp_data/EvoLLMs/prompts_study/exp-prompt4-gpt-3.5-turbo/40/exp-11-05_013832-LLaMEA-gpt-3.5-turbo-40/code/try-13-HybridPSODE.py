import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def cost_function(x):
            return func(x)

        def mutation(x, F):
            idxs = np.random.choice(range(self.dim), 3, replace=False)
            pbest = population[np.argmin([cost_function(p) for p in population])]
            mutant = x + F * (pbest - x)
            return np.clip(mutant, -5.0, 5.0)

        def crossover(x, mutant, CR):
            trial = np.copy(x)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    trial[i] = mutant[i]
            return trial

        def pso_update(x, pbest, w=0.5, c1=1.5, c2=1.5):
            v = np.random.uniform(-1, 1, self.dim)
            new_x = x + v
            new_x = np.clip(new_x, -5.0, 5.0)
            return new_x

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        F = 0.5
        CR = 0.9

        for _ in range(self.budget):
            for i in range(self.budget):
                x = population[i]
                F = 0.5 + 0.1 * np.random.randn()
                F = np.clip(F, 0.2, 0.8)

                mutant = mutation(x, F)
                trial = crossover(x, mutant, CR)
                trial = pso_update(x, trial)
                if cost_function(trial) < cost_function(x):
                    population[i] = trial

        best_idx = np.argmin([cost_function(x) for x in population])
        best_solution = population[best_idx]

        return best_solution