import numpy as np

class AdaptiveHybridEvolutionaryDifferentialHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def cost_function(x):
            return func(x)

        def adaptive_mutation(x, F, diversity):
            idxs = np.random.choice(range(self.dim), 3, replace=False)
            mutant = x + F * (population[np.random.randint(self.budget)] - population[np.random.randint(self.budget)]) * diversity
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

        # Adaptive mechanism for F based on diversity
        F_adapt = 0.5
        F_lower, F_upper = 0.2, 0.8

        for _ in range(self.budget):
            diversity = np.std(population)  # Calculate diversity
            for i in range(self.budget):
                x = population[i]
                F = F_adapt + 0.1 * np.random.randn()
                F = np.clip(F, F_lower, F_upper)

                mutant = adaptive_mutation(x, F, diversity)
                trial = crossover(x, mutant, CR)
                if cost_function(trial) < cost_function(x):
                    population[i] = trial
                else:
                    population[i] = harmonic_search(x)

        best_idx = np.argmin([cost_function(x) for x in population])
        best_solution = population[best_idx]

        return best_solution