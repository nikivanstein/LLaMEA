import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim, crossover_prob=0.9, scaling_factor=0.8):
        self.budget = budget
        self.dim = dim
        self.crossover_prob = crossover_prob
        self.scaling_factor = scaling_factor

    def __call__(self, func):
        pop_size = 10 * self.dim
        population = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        for _ in range(self.budget // pop_size):
            for i in range(pop_size):
                a, b, c = np.random.choice(population, 3, replace=False)
                mutant = np.clip(a + self.scaling_factor * (b - c), -5.0, 5.0)
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover_mask, mutant, population[i])
                f_trial = func(trial)
                if f_trial < fitness[i]:
                    population[i], fitness[i] = trial, f_trial
                
        return population[np.argmin(fitness)]