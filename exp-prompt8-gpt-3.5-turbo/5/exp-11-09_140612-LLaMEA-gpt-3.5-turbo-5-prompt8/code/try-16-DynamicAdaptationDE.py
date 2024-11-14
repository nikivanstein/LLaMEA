import numpy as np

class DynamicAdaptationDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        CR = 0.5
        F = 0.5
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        adapt_rate = 0.1
        
        for _ in range(self.budget):
            new_pop = np.copy(pop)
            for i in range(pop_size):
                candidates = np.random.choice(pop_size, size=3, replace=False)
                r1, r2, r3 = candidates
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                for j in range(self.dim):
                    if np.random.rand() > CR:
                        mutant[j] = pop[i][j]
                    F += np.random.uniform(-adapt_rate, adapt_rate)
                    F = max(0, min(1, F))
                new_fit = func(mutant)
                if new_fit < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = new_fit
                    CR += np.random.uniform(-adapt_rate, adapt_rate)
                    CR = max(0, min(1, CR))
                    
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        return best_solution