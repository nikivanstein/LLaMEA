import numpy as np

class CrowdedHybridEvolutionaryDifferentialHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def cost_function(x):
            return func(x)

        def mutation(x, F):
            idxs = np.random.choice(range(self.dim), 3, replace=False)
            diff = population[np.random.randint(self.budget)] - population[np.random.randint(self.budget)]
            mutant = x + F * diff
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

        # Crowding selection mechanism
        for _ in range(self.budget):
            scores = [cost_function(x) for x in population]
            sorted_indices = np.argsort(scores)
            for i in range(self.budget):
                idx = sorted_indices[i]
                x = population[idx]
                F = 0.5 + 0.3 * np.random.randn()
                F = np.clip(F, 0.2, 0.8)

                mutant = mutation(x, F)
                trial = crossover(x, mutant, CR)
                if cost_function(trial) < cost_function(x):
                    population[idx] = trial
                else:
                    population[idx] = harmonic_search(x)

        best_idx = np.argmin([cost_function(x) for x in population])
        best_solution = population[best_idx]

        return best_solution