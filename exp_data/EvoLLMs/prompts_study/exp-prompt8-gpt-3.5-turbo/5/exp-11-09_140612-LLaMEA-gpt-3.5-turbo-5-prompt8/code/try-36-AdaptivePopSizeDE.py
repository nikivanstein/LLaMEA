import numpy as np

class AdaptivePopSizeDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        pop_size = 10
        pop = np.random.uniform(-5.0, 5.0, (pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        CR = np.full(pop_size, 0.5)  # Dynamic adjustment of crossover rate
        F = np.full(pop_size, 0.5)   # Dynamic adjustment of mutation factor
        adapt_threshold = 0.1  # Threshold for adaptive population size adjustment

        for _ in range(self.budget):
            new_pop = np.copy(pop)
            for i in range(pop_size):
                candidates = np.random.choice(pop_size, size=3, replace=False)
                r1, r2, r3 = candidates
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])
                for j in range(self.dim):
                    if np.random.rand() > CR[i]:
                        mutant[j] = pop[i][j]
                new_fit = func(mutant)
                if new_fit < fitness[i]:
                    pop[i] = mutant
                    fitness[i] = new_fit
                    if np.random.rand() < adapt_threshold:
                        if new_fit < np.mean(fitness):
                            # Increase population size for better exploration
                            pop_size += 1
                            pop = np.vstack([pop, pop[i] + np.random.uniform(-0.1, 0.1, self.dim)])
                        else:
                            # Decrease population size for better exploitation
                            pop_size = max(1, pop_size - 1)
                            pop = pop[:-1]
                    
        best_idx = np.argmin(fitness)
        best_solution = pop[best_idx]
        return best_solution