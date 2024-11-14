import numpy as np

class EnhancedAdaptiveMutationOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 + int(10 * np.log(self.dim))
        self.bounds = (-5.0, 5.0)
        self.scale_factor = 0.82  # Slightly reduced for controlled exploration
        self.cross_prob = 0.92  # Slightly increased for better exploitation
        self.adaptation_rate = 0.04  # Adjusted adaptation rate
        self.exploit_factor = 0.22  # Adjusted for improved balance

    def __call__(self, func):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        func_evals = self.population_size

        while func_evals < self.budget:
            for i in range(self.population_size):
                if func_evals >= self.budget:
                    break

                idxs = list(range(self.population_size))
                idxs.remove(i)
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                adapt_factor = self.adaptation_rate * (np.random.rand() - 0.5)
                exploit_factor = self.exploit_factor * (0.5 - np.abs(np.mean(fitness) - fitness[i]) / np.std(fitness))
                mutant = a + (self.scale_factor + adapt_factor + exploit_factor) * (b - c)
                mutant = np.clip(mutant, self.bounds[0], self.bounds[1])

                trial = np.copy(population[i])
                crossover = np.random.rand(self.dim) < (self.cross_prob + adapt_factor * np.random.uniform(0.9, 1.05))
                trial[crossover] = mutant[crossover]

                trial_fitness = func(trial)
                func_evals += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

            self.scale_factor = np.random.uniform(0.72, 0.82)  # Narrowed range for more stability
            self.cross_prob = np.random.uniform(0.9, 0.98)  # Narrowed range for more consistent crossover

        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]