import numpy as np

class EnhancedPopulationDiversityEvolutionaryDifferentialHarmonySearch:
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

        def crowding_selection(population, cost_function):
            pop_size = len(population)
            sorted_indices = np.argsort([cost_function(x) for x in population])
            selected_indices = []
            while len(selected_indices) < pop_size:
                for idx in sorted_indices:
                    if idx not in selected_indices:
                        selected_indices.append(idx)
                    if len(selected_indices) == pop_size:
                        break
            return [population[idx] for idx in selected_indices]

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        F = 0.5
        CR = 0.9

        # Self-adaptive mechanism for F
        F_adapt = 0.5
        F_lower, F_upper = 0.2, 0.8

        for _ in range(self.budget):
            population = crowding_selection(population, cost_function)
            for i in range(self.budget):
                x = population[i]
                F = F_adapt + 0.1 * np.random.randn()
                F = np.clip(F, F_lower, F_upper)

                mutant = mutation(x, F)
                trial = crossover(x, mutant, CR)
                if cost_function(trial) < cost_function(x):
                    population[i] = trial

        best_idx = np.argmin([cost_function(x) for x in population])
        best_solution = population[best_idx]

        return best_solution