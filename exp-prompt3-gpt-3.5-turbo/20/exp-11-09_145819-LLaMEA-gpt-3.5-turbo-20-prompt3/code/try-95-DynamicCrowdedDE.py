import numpy as np
from scipy.spatial.distance import cdist

class DynamicCrowdedDE(DifferentialEvolution):
    def __init__(self, budget, dim, Cr=0.9, F=0.8, min_pop_size=10, max_pop_size=100, F_lb=0.2, F_ub=0.9, F_adapt=0.1, adapt_rate=0.05):
        super().__init__(budget, dim, Cr, F, min_pop_size)
        self.max_pop_size = max_pop_size
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.F_adapt = F_adapt
        self.adapt_rate = adapt_rate

    def __call__(self, func):
        def adapt_mutation_factor(F, fitness_progress):
            adapt_range = (1 - fitness_progress) * self.adapt_rate
            return np.clip(F + np.random.uniform(-adapt_range, adapt_range), self.F_lb, self.F_ub)

        def clip_to_bounds(x):
            return np.clip(x, self.lower_bound, self.upper_bound)

        def update_population(population, fitness_values):
            if len(population) < self.max_pop_size:
                new_individual = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
                population = np.vstack((population, new_individual))
                fitness_values = np.append(fitness_values, func(new_individual))
            return population, fitness_values

        population = np.random.uniform(self.lower_bound, self.upper_bound, (self.min_pop_size, self.dim))
        fitness_values = np.array([func(ind) for ind in population])
        evals = self.min_pop_size
        best_fitness = np.min(fitness_values)

        while evals < self.budget:
            population, fitness_values = update_population(population, fitness_values)

            for i in range(len(population)):
                a, b, c = np.random.choice(len(population), 3, replace=False)
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
            for i in range(len(population)):
                self.F = adapt_mutation_factor(self.F, (best_fitness - fitness_values[sorted_indices[i]]) / best_fitness)

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]

        return best_solution