import numpy as np

class AdaptiveChimeraAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rates = np.full((budget, dim), 0.5)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))
        for _ in range(self.budget):
            best_idx = np.argmin([func(ind) for ind in population])
            best_individual = population[best_idx]

            new_population = []
            for idx, ind in enumerate(population):
                mutation_prob = np.random.rand(self.dim)
                adapt_rates = np.clip(self.mutation_rates[idx] + 0.2 * (func(ind) < func(best_individual)), 0.1, 0.9)
                new_ind = np.where(mutation_prob < adapt_rates, ind + (best_individual - ind) * np.random.uniform(0.0, 1.0, self.dim), 
                                   np.where(mutation_prob < 0.6, ind + np.random.randn(self.dim), np.random.uniform(-5.0, 5.0, self.dim)))
                new_population.append(new_ind)
            population = np.array(new_population)
        best_idx = np.argmin([func(ind) for ind in population])
        return population[best_idx]