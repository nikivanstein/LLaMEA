import numpy as np

class EvoOrthogonalCauchyOptimization:
    def __init__(self, budget, dim, pop_size=30, f=0.5, cr=0.9, cauchy_scale=0.1):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f = f
        self.cr = cr
        self.cauchy_scale = cauchy_scale

    def __call__(self, func):
        def cauchy_mutation(individual):
            return np.clip(individual + self.cauchy_scale * np.tan(np.pi * (np.random.rand(self.dim) - 0.5)), -5.0, 5.0)

        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))

        for _ in range(self.budget):
            new_population = np.zeros_like(population)

            for i in range(self.pop_size):
                a, b, c = np.random.choice(self.pop_size, 3, replace=False)
                orthogonal_vector = np.linalg.qr(np.random.randn(self.dim, self.dim))[0][:, 0]
                mutant = np.clip(population[a] + self.f * (population[b] - population[c]), -5.0, 5.0)
                crossover = np.random.rand(self.dim) < self.cr
                trial = np.where(crossover, mutant, population[i])
                mutated_trial = cauchy_mutation(trial)

                if func(mutated_trial) < func(population[i]):
                    new_population[i] = mutated_trial
                else:
                    new_population[i] = population[i]

            population = new_population

        best_solution = population[np.argmin([func(ind) for ind in population])]
        return best_solution