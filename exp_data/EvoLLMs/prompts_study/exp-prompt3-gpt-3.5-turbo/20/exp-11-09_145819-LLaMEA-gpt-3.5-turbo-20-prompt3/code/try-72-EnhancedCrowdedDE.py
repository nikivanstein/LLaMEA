import numpy as np
from scipy.spatial.distance import cdist

class EnhancedCrowdedDE(DifferentialEvolution):
    def __init__(self, budget, dim, Cr=0.9, F=0.8, pop_size=50, F_lb=0.2, F_ub=0.9, F_adapt=0.1, adapt_rate=0.05, p=0.1):
        super().__init__(budget, dim, Cr, F, pop_size)
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.F_adapt = F_adapt
        self.adapt_rate = adapt_rate
        self.p = p

    def __call__(self, func):
        def adapt_mutation_factor(F, fitness_progress):
            adapt_range = (1 - fitness_progress) * self.adapt_rate
            return np.clip(F + np.random.uniform(-adapt_range, adapt_range), self.F_lb, self.F_ub)

        def create_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

        def clip_to_bounds(x):
            return np.clip(x, self.lower_bound, self.upper_bound)

        population = create_population()
        fitness_values = np.array([func(ind) for ind in population])
        evals = self.pop_size
        best_fitness = np.min(fitness_values)

        while evals < self.budget:
            new_population = []
            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                r = np.random.random(self.dim)
                mask = r < self.p
                mutant = clip_to_bounds(population[a] + self.F * (population[b] - population[c]) + np.random.uniform(-1, 1, self.dim) * mask)
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

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]

        return best_solution