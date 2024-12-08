import numpy as np

class DynamicSelectionMutationAlgorithm:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.mutation_rate = 0.1  # Initial mutation rate

    def diversity(self, population):
        return np.mean(np.std(population, axis=0))

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.budget, self.dim))

        for _ in range(self.budget):
            fitness = [func(ind) for ind in population]
            diversity_scores = [self.diversity(population)] * self.budget
            combined_scores = np.array(fitness) + 0.5 * np.array(diversity_scores)  # Dynamic selection
            best_idx = np.argmin(combined_scores)
            best_individual = population[best_idx]

            new_population = []
            for idx, ind in enumerate(population):
                mutation_rate = self.mutation_rate + 0.01 * (combined_scores[best_idx] - func(ind))
                mutated = ind + mutation_rate * np.random.randn(self.dim)
                mutated = np.clip(mutated, -5.0, 5.0)
                new_population.append(mutated)

            population = np.array(new_population)

        return best_individual