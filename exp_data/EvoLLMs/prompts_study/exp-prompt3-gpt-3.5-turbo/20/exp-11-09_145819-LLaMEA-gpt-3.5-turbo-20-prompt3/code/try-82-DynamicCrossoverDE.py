import numpy as np
from scipy.spatial.distance import cdist

class DynamicCrossoverDE(EnhancedCrowdedDE):
    def __init__(self, budget, dim, Cr=0.9, F=0.8, pop_size=50, F_lb=0.2, F_ub=0.9, F_adapt=0.1, adapt_rate=0.05, Cr_adapt=0.1, Cr_lb=0.1, Cr_ub=0.9):
        super().__init__(budget, dim, Cr, F, pop_size, F_lb, F_ub, F_adapt, adapt_rate)
        self.Cr_adapt = Cr_adapt
        self.Cr_lb = Cr_lb
        self.Cr_ub = Cr_ub

    def __call__(self, func):
        def adapt_crossover_rate(Cr, diversity_progress):
            adapt_range = (diversity_progress - 1) * self.Cr_adapt
            return np.clip(Cr + np.random.uniform(-adapt_range, adapt_range), self.Cr_lb, self.Cr_ub)

        population = create_population()
        fitness_values = np.array([func(ind) for ind in population])
        evals = self.pop_size
        best_fitness = np.min(fitness_values)

        while evals < self.budget:
            new_population = []
            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = clip_to_bounds(population[a] + self.F * (population[b] - population[c]))
                crossover = np.random.rand(self.dim) < self.Cr
                trial = population[i].copy()
                trial[crossover] = mutant[crossover]
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness

                best_fitness = min(best_fitness, trial_fitness)

            crowding_dist = cdist(population, population, 'euclidean')
            sorted_indices = np.argsort(crowding_dist.sum(axis=1))
            for i in range(self.pop_size):
                self.F = adapt_mutation_factor(self.F, (best_fitness - fitness_values[sorted_indices[i]]) / best_fitness)
                self.Cr = adapt_crossover_rate(self.Cr, np.std(population))

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]

        return best_solution