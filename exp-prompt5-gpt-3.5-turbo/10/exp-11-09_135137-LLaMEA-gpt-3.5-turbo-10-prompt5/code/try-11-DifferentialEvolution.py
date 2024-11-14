import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10
        self.mutation_factor = 0.5
        self.crossover_prob = 0.9

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                a, b, c = np.random.choice(self.population_size, 3, replace=False)
                mutant = pop[a] + self.mutation_factor * (pop[b] - pop[c])
                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover, mutant, pop[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
        return pop[np.argmin(fitness)]