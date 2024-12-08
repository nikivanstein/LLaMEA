import numpy as np

class EnhancedHybridDEGaussianOptimization:
    def __init__(self, budget, dim, pop_size=30, f=0.5, cr=0.9, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.sigma = sigma
        self.adaptive_sigma = 0.1  # initial mutation step size

    def __call__(self, func):
        def adaptive_gaussian_mutation(individual, sigma):
            return np.clip(individual + np.random.normal(0, sigma, self.dim), -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        for _ in range(self.budget):
            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = np.clip(population[a] + self.f * (population[b] - population[c]), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                mutated_trial = adaptive_gaussian_mutation(trial, self.adaptive_sigma)

                if func(mutated_trial) < func(population[i]):
                    new_population[i] = mutated_trial
                    self.adaptive_sigma *= 0.9  # decrease mutation step size if improvement
                else:
                    new_population[i] = population[i]
                    self.adaptive_sigma *= 1.1  # increase mutation step size if no improvement

            population = new_population

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution