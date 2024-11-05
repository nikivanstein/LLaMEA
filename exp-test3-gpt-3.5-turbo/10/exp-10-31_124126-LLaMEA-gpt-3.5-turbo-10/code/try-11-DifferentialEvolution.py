import numpy as np

class DifferentialEvolution:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        CR = 0.9
        F = 0.8
        bounds = [(-5.0, 5.0)] * self.dim
        best_solution = np.random.uniform(-5.0, 5.0, self.dim)
        best_fitness = func(best_solution)
        
        for _ in range(self.budget):
            pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
            for i in range(pop_size):
                a, b, c = np.random.choice(pop, 3, replace=False)
                mutant = np.clip(a + F * (b - c), -5.0, 5.0)
                jrand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])
                trial[jrand] = mutant[jrand]
                trial_fitness = func(trial)
                if trial_fitness < best_fitness:
                    best_solution = trial
                    best_fitness = trial_fitness
        return best_solution