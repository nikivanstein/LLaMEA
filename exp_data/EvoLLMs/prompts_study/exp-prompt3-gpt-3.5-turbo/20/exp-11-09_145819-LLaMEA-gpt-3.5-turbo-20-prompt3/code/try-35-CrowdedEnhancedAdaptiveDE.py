import numpy as np

class CrowdedEnhancedAdaptiveDE(EnhancedAdaptiveDE):
    def __init__(self, budget, dim, Cr=0.9, F=0.8, pop_size=50, F_lb=0.2, F_ub=0.9, F_adapt=0.1, adapt_rate=0.05, crowding_rate=0.3):
        super().__init__(budget, dim, Cr, F, pop_size, F_lb, F_ub, F_adapt, adapt_rate)
        self.crowding_rate = crowding_rate

    def __call__(self, func):
        def crowding_selection(population, fitness_values):
            sorted_indices = np.argsort(fitness_values)
            num_crowd = int(self.pop_size * self.crowding_rate)
            selected_indices = sorted_indices[:num_crowd]
            return population[selected_indices]

        population = create_population()
        fitness_values = np.array([func(ind) for ind in population])
        evals = self.pop_size
        best_fitness = np.min(fitness_values)

        while evals < self.budget:
            new_population = []
            crowded_population = crowding_selection(population, fitness_values)
            for i in range(self.pop_size):
                a, b, c = np.random.choice(crowded_population.shape[0], 3, replace=False)
                mutant = clip_to_bounds(crowded_population[a] + self.F * (crowded_population[b] - crowded_population[c]))
                crossover = np.random.rand(self.dim) < self.Cr
                trial = population[i].copy()
                trial[crossover] = mutant[crossover]
                trial_fitness = func(trial)
                evals += 1

                if trial_fitness < fitness_values[i]:
                    population[i] = trial
                    fitness_values[i] = trial_fitness

                best_fitness = min(best_fitness, trial_fitness)

            for i in range(self.pop_size):
                self.F = adapt_mutation_factor(self.F, (best_fitness - fitness_values[i]) / best_fitness)

        best_idx = np.argmin(fitness_values)
        best_solution = population[best_idx]

        return best_solution