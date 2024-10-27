import numpy as np

class EnhancedHybridDEGaussianOptimization:
    def __init__(self, budget, dim, pop_size=30, f=0.5, cr=0.9, sigma=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.sigma = sigma

    def __call__(self, func):
        def gaussian_mutation(individual):
            return np.clip(individual + np.random.normal(0, self.sigma, self.dim), -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        for _ in range(self.budget):
            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                mutant = np.clip(population[a] + self.f * (population[b] - population[c]), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                mutated_trial = gaussian_mutation(trial)

                new_population[i] = mutated_trial if func(mutated_trial) < func(population[i]) else population[i]

            population = new_population

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution