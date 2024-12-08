import numpy as np

class EnhancedDEGaussianOptimization:
    def __init__(self, budget, dim, pop_size=30, f=0.5, cr=0.9, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.sigma = sigma

    def __call__(self, func):
        def gaussian_mutation(individual, sigma):
            return np.clip(individual + np.random.normal(0, sigma, self.dim), -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        mutation_rates = np.full(self.pop_size, self.sigma)

        for _ in range(self.budget):
            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = np.clip(population[a] + self.f * (population[b] - population[c]), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                mutated_trial = gaussian_mutation(trial, mutation_rates[i])

                if func(mutated_trial) < func(population[i]):
                    new_population[i] = mutated_trial
                    if np.random.rand() < 0.2:
                        mutation_rates[i] *= 1.1  # Increase mutation rate
                else:
                    new_population[i] = population[i]
                    if np.random.rand() < 0.2:
                        mutation_rates[i] *= 0.9  # Decrease mutation rate

            population = new_population

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution