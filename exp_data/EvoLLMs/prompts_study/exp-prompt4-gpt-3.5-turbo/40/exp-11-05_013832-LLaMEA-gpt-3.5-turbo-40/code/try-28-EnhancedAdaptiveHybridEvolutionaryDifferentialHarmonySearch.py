import numpy as np

class EnhancedAdaptiveHybridEvolutionaryDifferentialHarmonySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        def cost_function(x):
            return func(x)

        def adaptive_mutation(x, F, diversity, best_individual, worst_individual):
            idxs = np.random.choice(range(self.dim), 3, replace=False)
            mutant = x + F * (best_individual - x) + F * (x - worst_individual) * diversity
            return np.clip(mutant, -5.0, 5.0)

        def crossover(x, mutant, CR):
            trial = np.copy(x)
            for i in range(self.dim):
                if np.random.rand() < CR:
                    trial[i] = mutant[i]
            return trial

        def perturbation(x, perturb_rate=0.1):
            perturbed_idx = np.random.choice(range(self.dim), int(self.dim * perturb_rate), replace=False)
            x[perturbed_idx] = np.random.uniform(-5.0, 5.0, len(perturbed_idx))
            return x

        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        F = 0.5
        CR = 0.9

        # Adaptive mechanism for F based on diversity
        F_adapt = 0.5
        F_lower, F_upper = 0.2, 0.8

        for _ in range(self.budget):
            diversity = np.std(population)  # Calculate diversity
            best_idx = np.argmin([cost_function(x) for x in population])
            worst_idx = np.argmax([cost_function(x) for x in population])
            best_individual = population[best_idx]
            worst_individual = population[worst_idx]

            for i in range(self.budget):
                x = population[i]
                F = F_adapt + 0.1 * np.random.randn()
                F = np.clip(F, F_lower, F_upper)

                mutant = adaptive_mutation(x, F, diversity, best_individual, worst_individual)
                trial = crossover(x, mutant, CR)
                if cost_function(trial) < cost_function(x):
                    population[i] = trial
                else:
                    population[i] = perturbation(x)

        best_idx = np.argmin([cost_function(x) for x in population])
        best_solution = population[best_idx]

        return best_solution