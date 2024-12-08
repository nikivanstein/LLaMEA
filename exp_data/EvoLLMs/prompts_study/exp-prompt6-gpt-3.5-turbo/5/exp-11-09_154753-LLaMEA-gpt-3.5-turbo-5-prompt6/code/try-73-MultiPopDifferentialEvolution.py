import numpy as np

class MultiPopDifferentialEvolution:
    def __init__(self, budget, dim, num_populations):
        self.budget = budget
        self.dim = dim
        self.num_populations = num_populations
        self.populations = [np.random.uniform(-5.0, 5.0, (budget, dim)) for _ in range(num_populations)]
    
    def __call__(self, func):
        scaling_factor = 0.8
        crossover_rate = 0.9
        diversity_factor = 0.5
        for _ in range(self.budget):
            for pop_idx in range(self.num_populations):
                population = self.populations[pop_idx]
                diversity = np.std(population)
                scaling_factor = 0.8 + diversity_factor * (diversity / 5.0)
                for i in range(self.budget):
                    a, b, c = np.random.choice(self.budget, 3, replace=False)
                    mutant_vector = population[a] + scaling_factor * (population[b] - population[c])
                    crossover_mask = np.random.rand(self.dim) < crossover_rate
                    trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                    if func(trial_vector) < func(population[i]):
                        population[i] = trial_vector
        best_individuals = [population[np.argmin([func(individual) for individual in population])] for population in self.populations]
        return best_individuals[np.argmin([func(individual) for individual in best_individuals])]