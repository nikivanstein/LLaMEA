import numpy as np

class DynamicAdaptiveHybridEvolutionaryDifferentialHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def cost_function(x):
            return func(x)

        def mutation(x, F):
            idxs = np.random.choice(range(self.dim), 3, replace=False)
            mutant = x + F * (population[np.random.randint(self.budget)] - population[np.random.randint(self.budget)])
            return np.clip(mutant, -5.0, 5.0)

        def crossover(x, mutant, CR):
            trial = np.copy(x)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    trial[i] = mutant[i]
            return trial

        def harmonic_search(x):
            new_x = np.copy(x)
            for i in range(self.dim):
                if np.random.rand() < 0.5:
                    new_x[i] = np.random.uniform(-5.0, 5.0)
            return new_x

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        F = 0.5
        CR = 0.9

        progress = 0
        max_progress = self.budget // 10
        F_min, F_max = 0.2, 0.8
        CR_min, CR_max = 0.6, 1.0

        for _ in range(self.budget):
            for i in range(self.budget):
                x = population[i]
                F = max(F_min, F_max - (F_max - F_min) * progress / max_progress)
                CR = min(CR_max, CR_min + (CR_max - CR_min) * progress / max_progress)
                
                mutant = mutation(x, F)
                trial = crossover(x, mutant, CR)
                if cost_function(trial) < cost_function(x):
                    population[i] = trial
                else:
                    population[i] = harmonic_search(x)
                
                progress += 1

        best_idx = np.argmin([cost_function(x) for x in population])
        best_solution = population[best_idx]

        return best_solution