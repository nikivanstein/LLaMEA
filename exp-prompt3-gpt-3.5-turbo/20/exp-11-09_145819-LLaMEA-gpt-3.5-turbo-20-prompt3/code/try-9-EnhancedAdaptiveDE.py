import numpy as np

class EnhancedAdaptiveDE(AdaptiveDE):
    def __init__(self, budget, dim, Cr=0.9, F=0.8, pop_size=50, F_lb=0.2, F_ub=0.9, F_adapt=0.1, adapt_rate=0.1):
        super().__init__(budget, dim, Cr, F, pop_size, F_lb, F_ub, F_adapt)
        self.adapt_rate = adapt_rate

    def __call__(self, func):
        def adapt_population_size(evals):
            return int(self.pop_size * (1 + self.adapt_rate * np.sin(np.pi * evals / self.budget)))

        def create_population():
            return np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))

        def clip_to_bounds(x):
            return np.clip(x, self.lower_bound, self.upper_bound)

        population = create_population()
        fitness_values = np.array([func(ind) for ind in population])
        evals = self.pop_size

        while evals < self.budget:
            self.pop_size = adapt_population_size(evals)
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

            best_idx = np.argmin(fitness_values)
            best_solution = population[best_idx]
            self.F = adapt_mutation_factor(self.F)

        return best_solution